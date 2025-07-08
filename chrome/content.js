// --- Кэши и Ограничения ---
const PROCESSED_MESSAGES_CACHE = new Map();
const CLASSIFIED_IMAGES_CACHE = new Map();
const MAX_PROCESSED_ITEMS = 500;

// --- Настройки (с значениями по умолчанию) ---
let settings = {
    excludedChannels: [],
    // Настройки рекламы
    enableAdClassification: true,
    adAnalysisMode: 'multimodal', // Новый режим
    adDisplayMode: 'highlight',
    adThreshold: 0.5,
    // Настройки NSFW
    enableNsfwClassification: true,
    nsfwDisplayMode: 'blur'
};

// --- Вспомогательная функция для логирования через background ---
function log(level, ...args) {
    try {
        chrome.runtime.sendMessage({
            action: "logFromContent",
            level: level.toUpperCase(),
            args: args
        });
    } catch (e) {
        // Fallback, если канал связи с background сломан (например, при перезагрузке расширения)
        console.log(`[LOG FALLBACK] ${level.toUpperCase()}:`, ...args);
    }
}

// --- Загрузка и обновление настроек ---
function loadAndUpdateSettings() {
    chrome.storage.local.get(null, (data) => {
        const oldSettings = JSON.stringify(settings);

        settings.excludedChannels = (data.excludedChannels || []).map(name => typeof name === 'string' ? name.toLowerCase() : '').filter(Boolean);
        settings.enableAdClassification = data.enableAdClassification !== false; // по умолчанию true
        settings.adAnalysisMode = data.adAnalysisMode || 'multimodal'; // НОВОЕ
        settings.adDisplayMode = data.displayMode || 'highlight';
        settings.adThreshold = data.threshold === undefined ? 0.5 : parseFloat(data.threshold);
        settings.enableNsfwClassification = data.enableNsfwClassification !== false; // по умолчанию true
        settings.nsfwDisplayMode = data.nsfwDisplayMode || 'blur';

        // Если что-то изменилось, запускаем полную перерисовку
        if (oldSettings !== JSON.stringify(settings)) {
            log('info', 'Settings changed. Re-processing content.', settings);
            
            // Сбрасываем кэши, чтобы всё пересчиталось с новыми настройками
            PROCESSED_MESSAGES_CACHE.clear();
            CLASSIFIED_IMAGES_CACHE.clear();

            // Сбрасываем все метки и стили со страницы
            document.querySelectorAll('.ad-classified, .nsfw-processed').forEach(el => {
                el.classList.remove('ad-classified', 'nsfw-processed', 'ad-excluded');
                el.style.cssText = ''; // Сброс инлайн стилей
            });
            document.querySelectorAll('.prediction-label, .nsfw-label, .nsfw-avatar-blur, .nsfw-avatar-border, .nsfw-avatar-hide').forEach(el => el.remove());

            // Запускаем обработку заново
            scheduleProcessNewContent();
        }
    });
}
chrome.storage.onChanged.addListener(loadAndUpdateSettings);

// --- Поиск элементов на странице ---
function getMessagesToProcess() {
    if (!settings.enableAdClassification) return [];
    const messages = [];
    document.querySelectorAll("div.bubble:not(.own):not(.ad-classified)").forEach(bubble => {
        const msgSpan = bubble.querySelector("span.translatable-message");
        const imageEl = bubble.querySelector('.message-photo-wrapper img, .message-video-wrapper video');
        const text = msgSpan ? msgSpan.textContent.trim() : '';
        const imageSrc = imageEl ? imageEl.src : null;
        if (text || imageSrc) {
            const messageId = bubble.dataset.mid || (text.slice(0, 30) + (imageSrc || Math.random()));
            messages.push({ id: messageId, text, imageSrc, bubbleElement: bubble });
        }
    });
    return messages;
}

function getImagesToProcess() {
    if (!settings.enableNsfwClassification) return [];
    const images = [];
    document.querySelectorAll('img:not(.nsfw-processed), video:not(.nsfw-processed)').forEach(el => {
        if (el.offsetParent !== null && el.src && (el.src.startsWith('http') || el.src.startsWith('blob:'))) {
            if (!CLASSIFIED_IMAGES_CACHE.has(el.src)) {
                images.push({ element: el, src: el.src });
            }
        }
    });
    return images;
}

// --- Логика классификации ---
async function processMessages(messages) {
    const currentChat = getCurrentChatName()?.toLowerCase();
    const isExcluded = settings.excludedChannels.includes(currentChat);

    for (const msg of messages) {
        msg.bubbleElement.classList.add('ad-classified');
        if (isExcluded) continue;

        if (PROCESSED_MESSAGES_CACHE.has(msg.id)) {
            applyAdStyle(msg.bubbleElement, PROCESSED_MESSAGES_CACHE.get(msg.id));
            continue;
        }

        try {
            const result = await chrome.runtime.sendMessage({
                action: "classifyMessage",
                text: msg.text,
                imageSrc: msg.imageSrc,
                mode: settings.adAnalysisMode // Передаем выбранный режим
            });

            if (result && !result.error) {
                PROCESSED_MESSAGES_CACHE.set(msg.id, result);
                applyAdStyle(msg.bubbleElement, result);
            } else if (result && result.error) {
                log('error', `API error for message ${msg.id}:`, result.error);
            }
        } catch (e) {
            log('error', `Failed to send/receive for message ${msg.id}:`, e.message);
        }
    }
}

async function processImages(images) {
    for (const img of images) {
        img.element.classList.add('nsfw-processed');

        if (CLASSIFIED_IMAGES_CACHE.has(img.src)) {
            applyNsfwStyle(img.element, CLASSIFIED_IMAGES_CACHE.get(img.src));
            continue;
        }

        try {
            const result = await chrome.runtime.sendMessage({
                action: "classifyImageNsfw",
                imageData: { type: 'url', url: img.src }
            });

            if (result && !result.error) {
                CLASSIFIED_IMAGES_CACHE.set(img.src, result);
                applyNsfwStyle(img.element, result);
            } else if (result && result.error) {
                log('error', `API error for image ${img.src}:`, result.error);
            }
        } catch (e) {
            log('error', `Failed to send/receive for image ${img.src}:`, e.message);
        }
    }
}

// --- Применение стилей ---
function applyAdStyle(bubble, result) {
    const oldLabel = bubble.querySelector('.prediction-label');
    if (oldLabel) oldLabel.remove();
    bubble.style.cssText = '';

    const isAd = result.prediction_prob_ad >= settings.adThreshold;

    const label = document.createElement("div");
    label.className = "prediction-label";
    label.textContent = `Ad: ${(result.prediction_prob_ad * 100).toFixed(1)}%`;
    label.classList.toggle('is-ad-positive', isAd);

    const container = bubble.querySelector('div.message') || bubble;
    container.prepend(label);

    if (isAd) {
        switch (settings.adDisplayMode) {
            case 'highlight':
                bubble.style.cssText = 'background-color: rgba(255, 204, 203, 0.2); border-left: 3px solid #ff7979;';
                break;
            case 'hide':
                bubble.style.display = 'none';
                break;
            case 'partial':
                bubble.style.opacity = '0.4';
                break;
            // 'prediction' case doesn't need extra styles
        }
    }
}

function applyNsfwStyle(element, result) {
    element.classList.remove('nsfw-avatar-blur', 'nsfw-avatar-border');
    const container = element.closest('.avatar') || element;
    container.classList.remove('nsfw-avatar-hide');
    container.style.display = '';
    element.style.display = '';

    if (result.is_nsfw) {
        switch (settings.nsfwDisplayMode) {
            case 'blur':
                element.classList.add('nsfw-avatar-blur');
                break;
            case 'hide':
                container.classList.add('nsfw-avatar-hide');
                break;
            case 'border':
                element.classList.add('nsfw-avatar-border');
                break;
            // 'nothing' case doesn't apply any style
        }
    }
}

// --- Основной цикл и наблюдатель ---
let processTimeoutId = null;
function scheduleProcessNewContent() {
    if (processTimeoutId) clearTimeout(processTimeoutId);
    processTimeoutId = setTimeout(processNewContent, 300);
}

function processNewContent() {
    if (settings.enableAdClassification) {
        const messages = getMessagesToProcess();
        if (messages.length > 0) processMessages(messages);
    }
    if (settings.enableNsfwClassification) {
        const images = getImagesToProcess();
        if (images.length > 0) processImages(images);
    }
}

function getCurrentChatName() {
    const headerElement = document.querySelector(".chat-info .person .content .user-title,.chat-info-container .chat-info .title,div.peer-title[data-peer-id]");
    if (headerElement) {
        return (headerElement.querySelector('span.peer-title') || headerElement).textContent.trim();
    }
    const hash = window.location.hash;
    return hash && hash.startsWith('#@') ? hash.substring(2) : null;
}

const observer = new MutationObserver((mutations) => {
    // Простая проверка, чтобы не вызывать обработку на каждое мелкое изменение
    for (const mutation of mutations) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            scheduleProcessNewContent();
            return;
        }
    }
});

function init() {
    loadAndUpdateSettings();
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    log('info', "Black-Fox Content Analyzer Loaded (v2.2).");
    setInterval(processNewContent, 3000); // Периодическая проверка на случай, если observer что-то упустит
}

if (document.readyState === "complete" || document.readyState === "interactive") {
    init();
} else {
    document.addEventListener("DOMContentLoaded", init);
}