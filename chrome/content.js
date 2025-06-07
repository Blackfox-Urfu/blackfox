// --- Кэши и Ограничения ---
const PROCESSED_MESSAGES_CACHE = new Map(); // Хранит { id: messageId, result: { prediction_prob_ad: number } }
const CLASSIFIED_AVATARS_CACHE = new Map(); // Хранит { src: avatarSrc, result: { is_nsfw: boolean, prediction_prob_nsfw?: number } }
const MAX_PROCESSED_ITEMS = 500; // Общее ограничение на хранение в каждом кэше

// --- Настройки (с значениями по умолчанию) ---
let settings = {
    excludedChannels: [],
    adDisplayMode: 'highlight',
    adThreshold: 0.5,
    classifyAvatarsEnabled: true,
    nsfwAvatarDisplayMode: 'blur'
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
        const oldAdDisplayMode = settings.adDisplayMode;
        const oldAdThreshold = settings.adThreshold;
        const oldExcludedChannelsString = settings.excludedChannels.join(',');
        const oldClassifyAvatarsEnabled = settings.classifyAvatarsEnabled;
        const oldNsfwAvatarDisplayMode = settings.nsfwAvatarDisplayMode;

        settings.excludedChannels = (data.excludedChannels || []).map(name => typeof name === 'string' ? name.toLowerCase() : '').filter(Boolean);
        settings.adDisplayMode = data.displayMode || 'highlight';
        settings.adThreshold = data.threshold === undefined ? 0.5 : parseFloat(data.threshold);
        settings.classifyAvatarsEnabled = data.classifyAvatarsEnabled === undefined ? true : data.classifyAvatarsEnabled;
        settings.nsfwAvatarDisplayMode = data.nsfwAvatarDisplayMode || 'blur';

        updateExcludeButtonVisibilityAndState();

        if (oldAdDisplayMode !== settings.adDisplayMode ||
            oldAdThreshold !== settings.adThreshold ||
            oldExcludedChannelsString !== settings.excludedChannels.join(',')) {
            
            document.querySelectorAll('div.bubble.ad-classified').forEach(bubble => {
                bubble.classList.remove('ad-classified', 'ad-excluded');
                bubble.style.backgroundColor = '';
                bubble.style.borderLeft = '';
                bubble.style.opacity = '';
                bubble.style.display = '';
                const predictionLabel = bubble.querySelector(".prediction-label");
                if (predictionLabel) predictionLabel.remove();
                const avatarLabel = bubble.querySelector(".avatar-prediction-label");
                if (avatarLabel) avatarLabel.remove();
            });
        }

        if (oldClassifyAvatarsEnabled !== settings.classifyAvatarsEnabled) {
            if (!settings.classifyAvatarsEnabled) {
                document.querySelectorAll('img.avatar-photo.nsfw-avatar-processed').forEach(img => {
                    resetNsfwAvatarStyle(img);
                    img.classList.remove('nsfw-avatar-processed', 'nsfw-marked', 'sfw-marked', 'classification-error', 'processing-error');
                });
                document.querySelectorAll('.avatar-prediction-label').forEach(label => label.remove());
            } else {
                document.querySelectorAll('img.avatar-photo.nsfw-avatar-processed').forEach(img => {
                     img.classList.remove('nsfw-avatar-processed');
                });
            }
        } else if (settings.classifyAvatarsEnabled && oldNsfwAvatarDisplayMode !== settings.nsfwAvatarDisplayMode) {
            document.querySelectorAll('img.avatar-photo.nsfw-avatar-processed').forEach(img => {
                img.classList.remove('nsfw-avatar-processed');
                resetNsfwAvatarStyle(img);
            });
             document.querySelectorAll('.avatar-prediction-label').forEach(label => label.remove());
        }
        
        scheduleProcessNewContent();
    });
}

chrome.storage.onChanged.addListener((changes, namespace) => {
    if (namespace === 'local') {
        let relevantChange = false;
        const relevantKeys = ['excludedChannels', 'displayMode', 'threshold', 'classifyAvatarsEnabled', 'nsfwAvatarDisplayMode'];
        for (let key in changes) {
            if (relevantKeys.includes(key)) {
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

// --- DOM Взаимодействие ---
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
        button.style.marginLeft = "10px";
        button.style.padding = "4px 8px";
        button.style.fontSize = "12px";
        button.style.border = "none";
        button.style.borderRadius = "4px";
        button.style.color = "white";
        button.style.cursor = "pointer";
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
            button.textContent = "Не фильтровать рекламу"; button.style.backgroundColor = "#f44336";
        }
    } else {
        button.style.display = "none";
    }
}

function getMessages() {
    const currentChat = getCurrentChatName();
    return Array.from(document.querySelectorAll("div.bubble:not(.own):not(.ad-classified) div.message > span.translatable-message"))
        .map(msgEl => {
            const bubble = msgEl.closest('div.bubble');
            if (!bubble) return null;
            // ВНИМАНИЕ: Использование Math.random() для messageId при отсутствии bubble.dataset.mid и bubble.offsetTop
            // сделает ID нестабильным и помешает кэшированию для таких сообщений.
            // Старайтесь полагаться на bubble.dataset.mid.
            const messageId = bubble.dataset.mid || (msgEl.textContent.slice(0,30) + '_' + msgEl.textContent.length + '_' + (bubble.offsetTop || Math.random().toString(36).substring(7)));
            return { id: messageId, element: msgEl, bubbleElement: bubble, text: msgEl.textContent.trim(), channelName: currentChat };
        }).filter(msg => msg && msg.text);
}

function getAvatars() {
    if (!settings.classifyAvatarsEnabled) return [];
    const avatarSelectors = ['img.avatar-photo'];
    const foundAvatars = [];
    document.querySelectorAll(avatarSelectors.join(', ')).forEach(img => {
        if (img.src && img.src !== 'about:blank' && !img.classList.contains('nsfw-avatar-processed')) {
            if (img.offsetParent !== null && (img.src.startsWith('blob:') || img.src.startsWith('http'))) {
                 foundAvatars.push({ element: img, src: img.src });
            }
        }
    });
    return foundAvatars;
}

// --- Применение стилей ---
function applyAdStyle(messageElement, predictionProbAd) {
    const bubble = messageElement.closest('div.bubble');
    if (!bubble) return;

    bubble.style.backgroundColor = '';
    bubble.style.borderLeft = '';
    bubble.style.opacity = '';
    bubble.style.display = '';

    if (predictionProbAd < settings.adThreshold) {
        return; 
    }

    switch (settings.adDisplayMode) {
        case 'highlight':
            bubble.style.backgroundColor = 'rgba(255, 204, 203, 0.2)';
            bubble.style.borderLeft = '3px solid #ff7979';
            break;
        case 'hide':
            bubble.style.display = 'none';
            break;
        case 'partial':
            bubble.style.opacity = '0.4';
            break;
    }
}

function resetNsfwAvatarStyle(imgElement) {
    imgElement.style.filter = '';
    imgElement.style.border = '';
    imgElement.style.opacity = '1';
    imgElement.classList.remove('nsfw-avatar-blur', 'nsfw-avatar-border');

    const parentAvatarDiv = imgElement.closest('.avatar');
    if (parentAvatarDiv) {
        parentAvatarDiv.style.display = '';
        parentAvatarDiv.classList.remove('nsfw-avatar-hide');
    } else {
        imgElement.style.display = '';
        imgElement.classList.remove('nsfw-avatar-hide');
    }
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
    imgElement.classList.add('nsfw-avatar-processed');
}

// --- Добавление/Обновление текстовых меток ---
function updateAdPredictionLabel(messageSpanElement, predictionProbAd) {
    const messageContentContainer = messageSpanElement.parentElement;
    if (!messageContentContainer) return;

    const currentChatName = getCurrentChatName() || '';
    let predictionElement = messageContentContainer.querySelector(".prediction-label");

    if (settings.excludedChannels.includes(currentChatName)) {
        if (predictionElement) predictionElement.remove();
        const avatarLabel = messageContentContainer.querySelector(".avatar-prediction-label");
        if (avatarLabel) avatarLabel.remove();
        return;
    }

    if (!predictionElement) {
        predictionElement = document.createElement("div");
        predictionElement.className = "prediction-label";
        messageContentContainer.insertBefore(predictionElement, messageSpanElement);
    }

    const probPercent = (predictionProbAd * 100).toFixed(1);
    predictionElement.textContent = `Реклама: ${probPercent}%`;
    predictionElement.classList.toggle('is-ad-positive', predictionProbAd >= settings.adThreshold);
    predictionElement.classList.toggle('is-ad-negative', predictionProbAd < settings.adThreshold);

    const avatarLabel = messageContentContainer.querySelector(".avatar-prediction-label");
    if (avatarLabel && predictionProbAd < settings.adThreshold) { // Если не реклама, удаляем метку аватара
        avatarLabel.remove();
    }
    // Если реклама, то addOrUpdateAvatarPredictionLabel позаботится о метке аватара
}

function addOrUpdateAvatarPredictionLabel(imgElement, classificationResult) {
    if (!settings.classifyAvatarsEnabled) return;

    const bubbleElement = imgElement.closest('div.bubble:not(.own)');
    if (bubbleElement) {
        const messageContentContainer = bubbleElement.querySelector('div.message');
        if (messageContentContainer) {
            const adLabelElement = messageContentContainer.querySelector(".prediction-label");
            let avatarLabel = messageContentContainer.querySelector(".avatar-prediction-label");

            if (adLabelElement && adLabelElement.classList.contains('is-ad-positive')) {
                if (!avatarLabel) {
                    avatarLabel = document.createElement('div');
                    avatarLabel.className = 'avatar-prediction-label';
                    adLabelElement.insertAdjacentElement('afterend', avatarLabel);
                }
                let nsfwScoreDisplay = "";
                if (classificationResult.prediction_prob_nsfw !== undefined && typeof classificationResult.prediction_prob_nsfw === 'number') {
                    nsfwScoreDisplay = ` (NSFW ${(classificationResult.prediction_prob_nsfw * 100).toFixed(1)}%)`;
                }
                const labelText = `Аватар: ${classificationResult.is_nsfw ? "NSFW" : "SFW"}${nsfwScoreDisplay}`;
                avatarLabel.textContent = labelText;
                avatarLabel.classList.toggle('is-nsfw-positive', classificationResult.is_nsfw);
                avatarLabel.classList.toggle('is-nsfw-negative', !classificationResult.is_nsfw);
            } else {
                if (avatarLabel) avatarLabel.remove();
            }
        }
    }
}

// --- Логика классификации ---
async function classifyMessages(messagesToClassify) {
    for (const msg of messagesToClassify) {
        if (!msg.bubbleElement || !msg.element) continue;

        if (settings.excludedChannels.includes(msg.channelName)) {
            applyAdStyle(msg.element, 0); // 0 -> сброс стилей
            updateAdPredictionLabel(msg.element, 0); // 0 -> удаление метки рекламы (и аватара)
            msg.bubbleElement.classList.add('ad-classified', 'ad-excluded');
            PROCESSED_MESSAGES_CACHE.delete(msg.id);
            continue;
        }
        
        if (PROCESSED_MESSAGES_CACHE.has(msg.id)) {
            const cachedData = PROCESSED_MESSAGES_CACHE.get(msg.id);
            if (cachedData && typeof cachedData.prediction_prob_ad !== 'undefined') {
                applyAdStyle(msg.element, cachedData.prediction_prob_ad);
                updateAdPredictionLabel(msg.element, cachedData.prediction_prob_ad);
                msg.bubbleElement.classList.add('ad-classified');
                continue;
            } else {
                PROCESSED_MESSAGES_CACHE.delete(msg.id);
            }
        }

        try {
            const response = await chrome.runtime.sendMessage({ action: "classify", text: msg.text });

            if (!response || typeof response !== "object") {
                console.error("CONTENT: Invalid response (message):", response, "Text:", msg.text.slice(0, 100));
                msg.bubbleElement.classList.add('ad-classified', 'classification-error');
                continue;
            }
            if (response.error) {
                console.error("CONTENT: API error (message):", response.error, "Text:", msg.text.slice(0, 100));
                msg.bubbleElement.classList.add('ad-classified', 'classification-error');
                continue;
            }
            
            const classificationResult = { prediction_prob_ad: response.prediction_prob_ad };
            PROCESSED_MESSAGES_CACHE.set(msg.id, classificationResult);
            applyAdStyle(msg.element, classificationResult.prediction_prob_ad);
            updateAdPredictionLabel(msg.element, classificationResult.prediction_prob_ad);
            msg.bubbleElement.classList.add('ad-classified');

        } catch (error) {
            console.error("CONTENT: Error during message classification call:", error, "Text:", msg.text.slice(0, 100));
            msg.bubbleElement.classList.add('ad-classified', 'processing-error');
        }
    }
}

async function classifyAvatars(avatarsToClassify) {
    if (!settings.classifyAvatarsEnabled) return;

    for (const avatar of avatarsToClassify) {
        if (!avatar.element) continue;

        if (CLASSIFIED_AVATARS_CACHE.has(avatar.src)) {
            const cachedResult = CLASSIFIED_AVATARS_CACHE.get(avatar.src);
             if (cachedResult && typeof cachedResult.is_nsfw !== 'undefined') {
                applyNsfwAvatarStyle(avatar.element, cachedResult.is_nsfw);
                addOrUpdateAvatarPredictionLabel(avatar.element, cachedResult);
            } else {
                 CLASSIFIED_AVATARS_CACHE.delete(avatar.src);
            }
            continue;
        }

        try {
            let imageDataPayload;
            let fileName = 'avatar.png';

            if (avatar.src.startsWith('blob:')) {
                const fetchResponse = await fetch(avatar.src);
                if (!fetchResponse.ok) throw new Error(`Fetch blob failed: ${fetchResponse.status} for ${avatar.src}`);
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
                console.error("CONTENT: Invalid response (avatar):", response, "URL:", avatar.src);
                avatar.element.classList.add('nsfw-avatar-processed', 'classification-error');
                continue;
            }
            if (response.error) {
                console.error("CONTENT: API error (avatar):", response.error, "URL:", avatar.src);
                avatar.element.classList.add('nsfw-avatar-processed', 'classification-error');
                // CLASSIFIED_AVATARS_CACHE.set(avatar.src, { is_nsfw: false, error: response.error }); // Опционально: кэшировать ошибку
                continue;
            }
            
            const classificationResult = {
                is_nsfw: response.is_nsfw,
                prediction_prob_nsfw: response.prediction_prob_nsfw
            };

            CLASSIFIED_AVATARS_CACHE.set(avatar.src, classificationResult);
            applyNsfwAvatarStyle(avatar.element, classificationResult.is_nsfw);
            addOrUpdateAvatarPredictionLabel(avatar.element, classificationResult);

        } catch (error) {
            console.error("CONTENT: Error processing/classifying avatar:", error, "URL:", avatar.src);
            avatar.element.classList.add('nsfw-avatar-processed', 'processing-error');
        }
    }
}

// --- Основной цикл и обработка ---
function cleanProcessedSets() {
    if (PROCESSED_MESSAGES_CACHE.size > MAX_PROCESSED_ITEMS) {
        const keysToDelete = Array.from(PROCESSED_MESSAGES_CACHE.keys()).slice(0, PROCESSED_MESSAGES_CACHE.size - MAX_PROCESSED_ITEMS + 100);
        keysToDelete.forEach(key => PROCESSED_MESSAGES_CACHE.delete(key));
    }
    if (CLASSIFIED_AVATARS_CACHE.size > MAX_PROCESSED_ITEMS) {
        const keysToDelete = Array.from(CLASSIFIED_AVATARS_CACHE.keys()).slice(0, CLASSIFIED_AVATARS_CACHE.size - MAX_PROCESSED_ITEMS + 100);
        keysToDelete.forEach(key => CLASSIFIED_AVATARS_CACHE.delete(key));
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

// --- Наблюдатель за изменениями DOM ---
const observer = new MutationObserver((mutationsList) => {
    let newContentPotentiallyAdded = false;
    for (const mutation of mutationsList) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            for (const node of mutation.addedNodes) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    if (node.matches && (node.matches('.bubble, .chat-list-item, .profile-view, .chat-info') || node.querySelector('.bubble, .avatar-photo'))) {
                        newContentPotentiallyAdded = true; break;
                    }
                }
            }
        }
        if (mutation.type === 'attributes' && mutation.attributeName === 'src' && mutation.target.matches && mutation.target.matches('img.avatar-photo')) {
            const imgTarget = mutation.target;
            if (imgTarget.classList.contains('nsfw-avatar-processed')) {
                resetNsfwAvatarStyle(imgTarget);
                imgTarget.classList.remove('nsfw-avatar-processed', 'nsfw-marked', 'sfw-marked', 'classification-error', 'processing-error');
                const bubble = imgTarget.closest('div.bubble:not(.own)');
                if (bubble) {
                    const avatarLabel = bubble.querySelector(".avatar-prediction-label");
                    if (avatarLabel) avatarLabel.remove();
                }
            }
            newContentPotentiallyAdded = true;
        }
        if (mutation.type === 'attributes' && mutation.attributeName === 'class' && mutation.target.matches && mutation.target.matches('div.bubble')) {
            const bubbleTarget = mutation.target;
            const messageId = bubbleTarget.dataset.mid; // Предполагаем, что ID есть в dataset.mid
            if (messageId && !bubbleTarget.classList.contains('ad-classified') && PROCESSED_MESSAGES_CACHE.has(messageId)) {
                // Если элемент потерял наш класс .ad-classified, но есть в кэше, это может быть признаком
                // того, что Telegram перерисовал его, и нужно восстановить нашу классификацию.
                // scheduleProcessNewContent() уже вызовется, если newContentPotentiallyAdded = true.
                // Можно просто убедиться, что getMessages() его подхватит, убрав класс .ad-classified,
                // что мы уже делаем в loadAndUpdateSettings при смене настроек.
                // Здесь можно просто установить флаг.
                newContentPotentiallyAdded = true;
            }
        }
        if (newContentPotentiallyAdded) break;
    }
    if (newContentPotentiallyAdded) {
        scheduleProcessNewContent();
    }
});

function startObserver() {
    const observeConfig = { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'class'] };
    const leftColumn = document.getElementById('LeftColumn');
    const middleColumn = document.getElementById('MiddleColumn');
    const rightColumn = document.getElementById('RightColumn');
    let observedSomething = false;
    if (leftColumn) { observer.observe(leftColumn, observeConfig); observedSomething = true; }
    if (middleColumn) { observer.observe(middleColumn, observeConfig); observedSomething = true; }
    if (rightColumn) { observer.observe(rightColumn, observeConfig); observedSomething = true; }
    if (!observedSomething) {
        observer.observe(document.body, observeConfig);
        console.log("CONTENT: Observer started on document.body.");
    } else {
        // console.log("CONTENT: Observer started on main columns.");
    }
}

// --- Периодические проверки и Инициализация ---
setInterval(processNewContent, 2000); // Дополнительный вызов на случай, если observer что-то пропустил
setInterval(updateExcludeButtonVisibilityAndState, 2500);

console.log("CONTENT: Ad and NSFW classifier script loaded (v.full.final).");

function initWhenReady() {
    if (document.readyState === "complete" || document.readyState === "interactive") {
        updateExcludeButtonVisibilityAndState();
        processNewContent();
        startObserver();
    } else {
        document.addEventListener("DOMContentLoaded", () => {
            updateExcludeButtonVisibilityAndState();
            processNewContent();
            startObserver();
        });
    }
}

initWhenReady();

// Напоминание: CSS стили должны быть определены в файле CSS вашего расширения
// или внедрены через JavaScript.
// Примерные CSS классы, которые используются в скрипте:
// .nsfw-avatar-blur, .nsfw-avatar-border, .nsfw-avatar-hide
// .prediction-label, .prediction-label.is-ad-positive, .prediction-label.is-ad-negative
// .avatar-prediction-label, .avatar-prediction-label.is-nsfw-positive, .avatar-prediction-label.is-nsfw-negative
