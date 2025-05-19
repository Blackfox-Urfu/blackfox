const PROCESSED_MESSAGES = new Set();
// ЗАМЕНА: Вместо Set будет Map для хранения результатов классификации аватарок
const CLASSIFIED_AVATARS_CACHE = new Map();
const MAX_PROCESSED_ITEMS = 500; // Общее ограничение на хранение ID/src

// --- Настройки (с значениями по умолчанию) ---
let settings = {
    excludedChannels: [],
    adDisplayMode: 'highlight',
    adThreshold: 0.5,
    classifyAvatarsEnabled: true,
    nsfwAvatarDisplayMode: 'border'
};

// --- Загрузка и обновление настроек ---
function loadAndUpdateSettings() {
    chrome.storage.local.get([
        'excludedChannels',
        'displayMode',
        'threshold',
        'classifyAvatarsEnabled',
        'nsfwAvatarDisplayMode'
    ], (data) => {
        settings.excludedChannels = (data.excludedChannels || []).map(name => typeof name === 'string' ? name.toLowerCase() : '').filter(Boolean);
        settings.adDisplayMode = data.displayMode || 'highlight';
        settings.adThreshold = data.threshold === undefined ? 0.5 : parseFloat(data.threshold);
        settings.classifyAvatarsEnabled = data.classifyAvatarsEnabled === undefined ? true : data.classifyAvatarsEnabled;
        settings.nsfwAvatarDisplayMode = data.nsfwAvatarDisplayMode || 'blur';

        updateExcludeButtonVisibilityAndState();

        if (!settings.classifyAvatarsEnabled) {
            document.querySelectorAll('img.avatar-photo.nsfw-avatar-processed').forEach(img => {
                resetNsfwAvatarStyle(img);
                img.classList.remove('nsfw-avatar-processed', 'nsfw-marked', 'sfw-marked');
            });
            // Очищаем кэш, чтобы при включении заново обработать и классифицировать
            CLASSIFIED_AVATARS_CACHE.clear();
            document.querySelectorAll('.avatar-prediction-label').forEach(label => label.remove());
        } else {
            // Переприменить стили и метки, если режим отображения NSFW изменился или классификация была только что включена
            // Это потребует перебора всех видимых аватарок и применения к ним данных из кеша.
            // Либо, более простой подход: при изменении настроек сбросить status 'nsfw-avatar-processed' у всех
            // и позволить processNewContent() их заново "обработать" (взяв данные из кеша или классифицировав).
            document.querySelectorAll('img.avatar-photo.nsfw-avatar-processed').forEach(img => {
                 img.classList.remove('nsfw-avatar-processed'); // Позволит их снова найти в getAvatars
            });
            // Таким образом, следующий вызов processNewContent подхватит их.
        }
    });
}

chrome.storage.onChanged.addListener((changes, namespace) => {
    if (namespace === 'local') {
        let relevantChange = false;
        for (let key in changes) {
            if (settings.hasOwnProperty(key) ||
                (key === 'displayMode' && 'adDisplayMode' in settings) ||
                (key === 'threshold' && 'adThreshold' in settings)
            ) {
                relevantChange = true;
                break;
            }
        }
        if (relevantChange) {
            loadAndUpdateSettings();
        }
    }
});

loadAndUpdateSettings();

// --- DOM Взаимодействие (без изменений) ---
function getChatHeaderElement() {
    return document.querySelector(".chat-info .person .content .user-title") ||
           document.querySelector(".chat-info-container .chat-info .title") ||
           document.querySelector('div.peer-title[data-peer-id]');
}
function getCurrentChatName() {
    const headerElement = getChatHeaderElement();
    if (headerElement) {
        const peerTitleSpan = headerElement.matches('span.peer-title') ? headerElement : headerElement.querySelector('span.peer-title');
        if (peerTitleSpan && peerTitleSpan.textContent) return peerTitleSpan.textContent.trim().toLowerCase();
        if (headerElement.textContent) return headerElement.textContent.trim().toLowerCase();
    }
    const hash = window.location.hash;
    if (hash && hash.startsWith('#@')) return hash.substring(2).toLowerCase();
    return null;
}
function updateExcludeButtonVisibilityAndState() {
    const chatTitleContainer = getChatHeaderElement()?.closest('.user-title') || getChatHeaderElement()?.closest('.title');
    if (!chatTitleContainer) {
        const existingButton = document.getElementById("excludeChannelBtn");
        if (existingButton) existingButton.remove();
        return;
    }
    let button = document.getElementById("excludeChannelBtn");
    if (!button) {
        button = document.createElement("button");
        button.id = "excludeChannelBtn";
        chatTitleContainer.appendChild(button);
        button.onclick = function () {
            const currentChat = getCurrentChatName();
            if (!currentChat) return;
            const isExcluded = settings.excludedChannels.includes(currentChat);
            let updatedExcludedChannels = isExcluded ? settings.excludedChannels.filter(cn => cn !== currentChat) : [...settings.excludedChannels, currentChat];
            chrome.storage.local.set({ excludedChannels: updatedExcludedChannels });
        };
    }
    const currentChat = getCurrentChatName();
    if (currentChat) {
        button.style.display = "inline-block";
        if (settings.excludedChannels.includes(currentChat)) {
            button.textContent = "Фильтровать рекламу"; button.style.backgroundColor = "#4CAF50";
        } else {
            button.textContent = "Не фильтровать рекламу"; button.style.backgroundColor = "#ff5c5c";
        }
    } else {
        button.style.display = "none";
    }
}
function getMessages() {
    const currentChat = getCurrentChatName();
    return Array.from(document.querySelectorAll("div.bubble:not(.own) div.message > span.translatable-message"))
        .map(msgEl => {
            const bubble = msgEl.closest('div.bubble');
            if (!bubble) return null;
            const messageId = bubble.dataset.mid || (msgEl.textContent.slice(0,30) + '_' + msgEl.textContent.length);
            if (PROCESSED_MESSAGES.has(messageId)) return null;
            return { id: messageId, element: msgEl, text: msgEl.textContent.trim(), channelName: currentChat };
        }).filter(msg => msg && msg.text);
}

function getAvatars() {
    if (!settings.classifyAvatarsEnabled) return [];
    const avatarSelectors = ['img.avatar-photo'];
    const foundAvatars = [];
    document.querySelectorAll(avatarSelectors.join(', ')).forEach(img => {
        // ИЗМЕНЕНИЕ: Проверяем только на 'nsfw-avatar-processed'. Факт наличия в кеше проверим позже.
        if (img.src && img.src !== 'about:blank' && !img.classList.contains('nsfw-avatar-processed')) {
            if (img.src.startsWith('blob:') || img.src.startsWith('http')) {
                if (img.offsetParent !== null) {
                    foundAvatars.push({ element: img, src: img.src });
                }
            }
        }
    });
    return foundAvatars;
}

// --- Применение стилей (applyAdStyle, resetNsfwAvatarStyle, applyNsfwAvatarStyle - без изменений) ---
function applyAdStyle(messageElement, predictionProbAd) {
    const bubbleContent = messageElement.closest('div.bubble-content');
    if (!bubbleContent) return;
    const bubble = bubbleContent.closest('div.bubble');
    if (!bubble) return;
    bubble.style.backgroundColor = ''; bubble.style.borderLeft = ''; bubble.style.opacity = ''; bubble.style.display = '';
    if (predictionProbAd < settings.adThreshold) return;
    switch (settings.adDisplayMode) {
        case 'highlight': bubble.style.backgroundColor = 'rgba(255, 204, 203, 0.2)'; bubble.style.borderLeft = '3px solid #ff7979'; break;
        case 'hide': bubble.style.display = 'none'; break;
        case 'partial': bubble.style.opacity = '0.4'; break;
    }
}
function resetNsfwAvatarStyle(imgElement) {
    imgElement.style.filter = ''; imgElement.style.border = ''; imgElement.style.opacity = '1';
    imgElement.classList.remove('nsfw-avatar-blur', 'nsfw-avatar-border');
    const parentAvatarDiv = imgElement.closest('.avatar');
    if (parentAvatarDiv) { parentAvatarDiv.style.display = ''; parentAvatarDiv.classList.remove('nsfw-avatar-hide'); }
    else { imgElement.style.display = ''; imgElement.classList.remove('nsfw-avatar-hide'); }
}
function applyNsfwAvatarStyle(imgElement, isNsfw) {
    resetNsfwAvatarStyle(imgElement);
    imgElement.classList.remove('nsfw-marked', 'sfw-marked');
    if (isNsfw) {
        imgElement.classList.add('nsfw-marked');
        switch (settings.nsfwAvatarDisplayMode) {
            case 'blur': imgElement.classList.add('nsfw-avatar-blur'); break;
            case 'hide':
                const parentAvatarDiv = imgElement.closest('.avatar');
                if (parentAvatarDiv) parentAvatarDiv.classList.add('nsfw-avatar-hide');
                else imgElement.classList.add('nsfw-avatar-hide');
                break;
            case 'border': imgElement.classList.add('nsfw-avatar-border'); break;
        }
    } else {
        imgElement.classList.add('sfw-marked');
    }
    imgElement.classList.add('nsfw-avatar-processed'); // Помечаем элемент как обработанный (стили применены)
}

// --- Добавление/Обновление текстовой метки для аватара ---
function addOrUpdateAvatarPredictionLabel(imgElement, classificationResult) {
    const bubbleElement = imgElement.closest('div.bubble:not(.own)');
    if (bubbleElement && settings.classifyAvatarsEnabled) {
        const messageContentContainer = bubbleElement.querySelector('div.message');
        if (messageContentContainer) {
            const adPredictionLabel = messageContentContainer.querySelector(".prediction-label");
            if (adPredictionLabel) { // Только если есть метка рекламы
                let avatarLabel = messageContentContainer.querySelector(".avatar-prediction-label");
                if (!avatarLabel) {
                    avatarLabel = document.createElement('div');
                    avatarLabel.className = 'avatar-prediction-label';
                    adPredictionLabel.insertAdjacentElement('afterend', avatarLabel);
                }
                let nsfwScoreDisplay = "";
                if (classificationResult.prediction_prob_nsfw !== undefined && typeof classificationResult.prediction_prob_nsfw === 'number') {
                    nsfwScoreDisplay = ` (NSFW ${(classificationResult.prediction_prob_nsfw * 100).toFixed(1)}%)`;
                }
                const labelText = `Аватар: ${classificationResult.is_nsfw ? "NSFW" : "SFW"}${nsfwScoreDisplay}`;
                avatarLabel.textContent = labelText;
                avatarLabel.classList.toggle('is-nsfw-positive', classificationResult.is_nsfw);
                avatarLabel.classList.toggle('is-nsfw-negative', !classificationResult.is_nsfw);
            }
        }
    }
}


// --- Логика классификации (classifyMessages - без изменений) ---
async function classifyMessages(messagesToClassify) {
    for (const msg of messagesToClassify) {
        if (settings.excludedChannels.includes(msg.channelName)) {
            PROCESSED_MESSAGES.add(msg.id); continue;
        }
        try {
            const response = await chrome.runtime.sendMessage({ action: "classify", text: msg.text });
            if (!response || typeof response !== "object") { PROCESSED_MESSAGES.add(msg.id); continue; }
            if (response.error) { PROCESSED_MESSAGES.add(msg.id); continue; }
            applyAdStyle(msg.element, response.prediction_prob_ad);
            let predictionElement = msg.element.parentElement.querySelector(".prediction-label");
            if (!predictionElement) {
                predictionElement = document.createElement("div");
                predictionElement.className = "prediction-label";
                if(msg.element.parentElement) msg.element.parentElement.insertBefore(predictionElement, msg.element);
            }
            const probPercent = (response.prediction_prob_ad * 100).toFixed(1);
            predictionElement.textContent = `Реклама: ${probPercent}%`;
            predictionElement.classList.toggle('is-ad-positive', response.prediction_prob_ad >= settings.adThreshold);
            predictionElement.classList.toggle('is-ad-negative', response.prediction_prob_ad < settings.adThreshold);
            PROCESSED_MESSAGES.add(msg.id);
        } catch (error) { PROCESSED_MESSAGES.add(msg.id); }
    }
}

// --- ИЗМЕНЕННАЯ ЛОГИКА classifyAvatars ---
async function classifyAvatars(avatarsToClassify) {
    if (!settings.classifyAvatarsEnabled) return;

    for (const avatar of avatarsToClassify) {
        // avatar.element УЖЕ не имеет 'nsfw-avatar-processed' благодаря getAvatars()
        // Теперь проверяем кэш
        if (CLASSIFIED_AVATARS_CACHE.has(avatar.src)) {
            const cachedResult = CLASSIFIED_AVATARS_CACHE.get(avatar.src);
            applyNsfwAvatarStyle(avatar.element, cachedResult.is_nsfw);
            addOrUpdateAvatarPredictionLabel(avatar.element, cachedResult);
            // avatar.element.classList.add('nsfw-avatar-processed'); // applyNsfwAvatarStyle уже делает это
            continue; // Переходим к следующему аватару
        }

        // Если в кэше нет, классифицируем
        try {
            let imageDataPayload;
            let fileName = 'avatar.png';
            if (avatar.src.startsWith('blob:')) {
                const fetchResponse = await fetch(avatar.src);
                if (!fetchResponse.ok) throw new Error(`Fetch blob failed: ${fetchResponse.status}`);
                const blobData = await fetchResponse.blob();
                const arrayBuffer = await blobData.arrayBuffer();
                if (blobData.type && blobData.type.startsWith('image/')) {
                    const ext = blobData.type.split('/')[1];
                    if (ext && ['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext.toLowerCase())) fileName = `avatar.${ext}`;
                }
                imageDataPayload = { type: 'arrayBuffer', buffer: Array.from(new Uint8Array(arrayBuffer)), mimeType: blobData.type || 'application/octet-stream', fileName: fileName };
            } else {
                try {
                    const urlPath = new URL(avatar.src).pathname;
                    const extension = urlPath.substring(urlPath.lastIndexOf('.') + 1);
                    if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(extension.toLowerCase())) fileName = `avatar.${extension}`;
                } catch (e) { /* ignore */ }
                imageDataPayload = { type: 'url', url: avatar.src, fileName: fileName };
            }

            const response = await chrome.runtime.sendMessage({ action: "classifyAvatar", imageData: imageDataPayload });

            if (!response || typeof response !== "object") {
                // Помечаем элемент, чтобы не пытаться снова с этим элементом, но не кэшируем как "неудачный" URL
                avatar.element.classList.add('nsfw-avatar-processed', 'classification-error');
                console.error("CONTENT: Invalid response (avatar):", response, "URL:", avatar.src);
                continue;
            }
            if (response.error) {
                avatar.element.classList.add('nsfw-avatar-processed', 'classification-error');
                console.error("CONTENT: API error (avatar):", response.error, "URL:", avatar.src);
                // Здесь можно кэшировать ошибку для URL, чтобы не долбить API по этому URL
                // CLASSIFIED_AVATARS_CACHE.set(avatar.src, { error: response.error, is_nsfw: false });
                continue;
            }
            
            // Успешная классификация
            const classificationResult = {
                is_nsfw: response.is_nsfw,
                prediction_prob_nsfw: response.prediction_prob_nsfw
                // можно добавить и другие поля из response, если они нужны
            };

            CLASSIFIED_AVATARS_CACHE.set(avatar.src, classificationResult); // Кэшируем результат
            applyNsfwAvatarStyle(avatar.element, classificationResult.is_nsfw);
            addOrUpdateAvatarPredictionLabel(avatar.element, classificationResult);
            // avatar.element.classList.add('nsfw-avatar-processed'); // applyNsfwAvatarStyle уже делает это

        } catch (error) {
            console.error("CONTENT: Error processing/classifying avatar:", error, "URL:", avatar.src);
            // Помечаем только элемент, чтобы не пытаться снова с этим экземпляром, но URL не кэшируем как ошибочный глобально
            avatar.element.classList.add('nsfw-avatar-processed', 'processing-error');
        }
    }
}

// --- Основной цикл и обработка ---
function cleanProcessedSets() {
    if (PROCESSED_MESSAGES.size > MAX_PROCESSED_ITEMS) {
        const messagesToDelete = Array.from(PROCESSED_MESSAGES).slice(0, PROCESSED_MESSAGES.size - MAX_PROCESSED_ITEMS + 100);
        messagesToDelete.forEach(id => PROCESSED_MESSAGES.delete(id));
    }
    // Очистка кэша аватарок, если он слишком разросся
    if (CLASSIFIED_AVATARS_CACHE.size > MAX_PROCESSED_ITEMS) {
        // Удаляем самые старые записи. Map сохраняет порядок вставки.
        const keysToDelete = Array.from(CLASSIFIED_AVATARS_CACHE.keys()).slice(0, CLASSIFIED_AVATARS_CACHE.size - MAX_PROCESSED_ITEMS + 100);
        keysToDelete.forEach(key => CLASSIFIED_AVATARS_CACHE.delete(key));
        // console.log(`CONTENT: Avatar cache cleaned. Size: ${CLASSIFIED_AVATARS_CACHE.size}`);
    }
}

let processTimeoutId = null;
function scheduleProcessNewContent() {
    if (processTimeoutId) clearTimeout(processTimeoutId);
    processTimeoutId = setTimeout(processNewContent, 300);
}

function processNewContent() {
    const messages = getMessages();
    if (messages.length > 0) {
        classifyMessages(messages);
    }
    if (settings.classifyAvatarsEnabled) {
        const avatars = getAvatars();
        if (avatars.length > 0) {
            classifyAvatars(avatars);
        }
    }
    cleanProcessedSets();
}

// --- Наблюдатель за изменениями DOM (без изменений) ---
const observer = new MutationObserver((mutationsList) => {
    let newContentPotentiallyAdded = false;
    for (const mutation of mutationsList) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            for (const node of mutation.addedNodes) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    if (node.matches && (node.matches('.bubble, .chat-list-item, .profile-view') || node.querySelector('.bubble, .avatar-photo'))) {
                        newContentPotentiallyAdded = true; break;
                    }
                }
            }
        }
        if (mutation.type === 'attributes' && mutation.attributeName === 'src' && mutation.target.matches && mutation.target.matches('img.avatar-photo')) {
            // Если src изменился, и это аватарка, которая уже была "обработана" (например, заблюрена),
            // нужно снять с нее класс 'nsfw-avatar-processed', чтобы она снова попала в getAvatars для новой обработки/проверки кеша
            if (mutation.target.classList.contains('nsfw-avatar-processed')) {
                mutation.target.classList.remove('nsfw-avatar-processed');
                 // Можно также удалить ее специфичные стили, если они были установлены напрямую, а не классами
                resetNsfwAvatarStyle(mutation.target);
                // И удалить текстовую метку, если она была рядом
                const bubble = mutation.target.closest('div.bubble:not(.own)');
                if (bubble) {
                    const avatarLabel = bubble.querySelector(".avatar-prediction-label");
                    if (avatarLabel) avatarLabel.remove();
                }
            }
            newContentPotentiallyAdded = true;
        }
        if (newContentPotentiallyAdded) break;
    }
    if (newContentPotentiallyAdded) {
        scheduleProcessNewContent();
    }
});
const leftColumn = document.getElementById('LeftColumn');
const middleColumn = document.getElementById('MiddleColumn');
const rightColumn = document.getElementById('RightColumn');
if (leftColumn) observer.observe(leftColumn, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
if (middleColumn) observer.observe(middleColumn, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
if (rightColumn) observer.observe(rightColumn, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
if (!leftColumn && !middleColumn && !rightColumn) {
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
}

setInterval(processNewContent, 2000);
setInterval(updateExcludeButtonVisibilityAndState, 2500);
console.log("CONTENT: Скрипт content.js загружен и активен (с кэшированием аватарок).");

function initWhenReady() {
    if (document.readyState === "complete" || document.readyState === "interactive") {
        updateExcludeButtonVisibilityAndState(); processNewContent();
    } else {
        document.addEventListener("DOMContentLoaded", () => { updateExcludeButtonVisibilityAndState(); processNewContent(); });
    }
}
initWhenReady();