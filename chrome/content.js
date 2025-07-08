// --- Кэши и Ограничения ---
const PROCESSED_MESSAGES_CACHE = new Map();
const CLASSIFIED_IMAGES_CACHE = new Map();
const MAX_PROCESSED_ITEMS = 500;

let settings = { excludedChannels: [], enableAdClassification: true, adDisplayMode: 'highlight', adThreshold: 0.5, enableNsfwClassification: true, nsfwDisplayMode: 'blur' };

function log(level, ...args) {
    try { chrome.runtime.sendMessage({ action: "logFromContent", level: level.toUpperCase(), args: args }); } 
    catch (e) { console.log(`[LOG FALLBACK] ${level.toUpperCase()}:`, ...args); }
}

function loadAndUpdateSettings() {
    chrome.storage.local.get(null, (data) => {
        const oldSettings = JSON.stringify(settings);
        settings.excludedChannels = (data.excludedChannels || []).map(name => typeof name === 'string' ? name.toLowerCase() : '').filter(Boolean);
        settings.enableAdClassification = data.enableAdClassification === undefined ? true : data.enableAdClassification;
        settings.adDisplayMode = data.displayMode || 'highlight';
        settings.adThreshold = data.threshold === undefined ? 0.5 : parseFloat(data.threshold);
        settings.enableNsfwClassification = data.enableNsfwClassification === undefined ? true : data.enableNsfwClassification;
        settings.nsfwDisplayMode = data.nsfwAvatarDisplayMode || 'blur';

        if (oldSettings !== JSON.stringify(settings)) {
            log('info', 'Settings changed. Re-processing content.', settings);
            PROCESSED_MESSAGES_CACHE.clear();
            CLASSIFIED_IMAGES_CACHE.clear();
            document.querySelectorAll('.ad-classified, .nsfw-processed').forEach(el => el.classList.remove('ad-classified', 'nsfw-processed'));
            document.querySelectorAll('.prediction-label, .nsfw-label, .nsfw-avatar-blur, .nsfw-avatar-border').forEach(el => el.remove());
            scheduleProcessNewContent();
        }
    });
}
chrome.storage.onChanged.addListener(loadAndUpdateSettings);

function getMessagesToProcess() {
    if (!settings.enableAdClassification) return [];
    const messages = [];
    document.querySelectorAll("div.bubble:not(.own):not(.ad-classified)").forEach(bubble => {
        const msgSpan = bubble.querySelector("span.translatable-message");
        const imageEl = bubble.querySelector('.message-photo-wrapper img, .message-video-wrapper video');
        const text = msgSpan ? msgSpan.textContent.trim() : '';
        const imageSrc = imageEl ? imageEl.src : null;
        if (text || imageSrc) {
            const messageId = bubble.dataset.mid || (text.slice(0, 30) + (imageSrc || ''));
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

async function processMessages(messages) {
    const currentChat = getCurrentChatName()?.toLowerCase();
    const isExcluded = settings.excludedChannels.includes(currentChat);

    for (const msg of messages) {
        msg.bubbleElement.classList.add('ad-classified');
        if (isExcluded) continue;
        if (PROCESSED_MESSAGES_CACHE.has(msg.id)) { applyAdStyle(msg.bubbleElement, PROCESSED_MESSAGES_CACHE.get(msg.id)); continue; }
        try {
            const result = await chrome.runtime.sendMessage({ action: "classifyMessage", text: msg.text, imageSrc: msg.imageSrc });
            if (result && !result.error) { PROCESSED_MESSAGES_CACHE.set(msg.id, result); applyAdStyle(msg.bubbleElement, result); } 
            else if (result.error) { log('error', `API error for message ${msg.id}:`, result.error); }
        } catch (e) { log('error', `Failed to classify message ${msg.id}:`, e.message); }
    }
}

async function processImages(images) {
    for (const img of images) {
        img.element.classList.add('nsfw-processed');
        if (CLASSIFIED_IMAGES_CACHE.has(img.src)) { applyNsfwStyle(img.element, CLASSIFIED_IMAGES_CACHE.get(img.src)); continue; }
        try {
            const result = await chrome.runtime.sendMessage({ action: "classifyImageNsfw", imageData: { type: 'url', url: img.src } });
            if (result && !result.error) { CLASSIFIED_IMAGES_CACHE.set(img.src, result); applyNsfwStyle(img.element, result); } 
            else if (result.error) { log('error', `API error for image ${img.src}:`, result.error); }
        } catch (e) { log('error', `Failed to classify image ${img.src}:`, e.message); }
    }
}

function applyAdStyle(bubble, result) {
    const oldLabel = bubble.querySelector('.prediction-label');
    if (oldLabel) oldLabel.remove();
    bubble.style = {};
    const isAd = result.prediction_prob_ad >= settings.adThreshold;
    const label = document.createElement("div");
    label.className = "prediction-label";
    label.textContent = `Ad: ${(result.prediction_prob_ad * 100).toFixed(1)}%`;
    label.classList.toggle('is-ad-positive', isAd);
    (bubble.querySelector('div.message') || bubble).prepend(label);
    if (isAd) {
        switch (settings.adDisplayMode) {
            case 'highlight': bubble.style.cssText = 'background-color: rgba(255, 204, 203, 0.2); border-left: 3px solid #ff7979;'; break;
            case 'hide': bubble.style.display = 'none'; break;
            case 'partial': bubble.style.opacity = '0.4'; break;
        }
    }
}

function applyNsfwStyle(element, result) {
    element.classList.remove('nsfw-avatar-blur', 'nsfw-avatar-border');
    (element.closest('.avatar') || element).classList.remove('nsfw-avatar-hide');
    if (result.is_nsfw) {
        switch (settings.nsfwDisplayMode) {
            case 'blur': element.classList.add('nsfw-avatar-blur'); break;
            case 'hide': (element.closest('.avatar') || element).classList.add('nsfw-avatar-hide'); break;
            case 'border': element.classList.add('nsfw-avatar-border'); break;
        }
    }
}

let processTimeoutId = null;
function scheduleProcessNewContent() { if (processTimeoutId) clearTimeout(processTimeoutId); processTimeoutId = setTimeout(processNewContent, 300); }
function processNewContent() {
    if (settings.enableAdClassification) { const messages = getMessagesToProcess(); if (messages.length > 0) processMessages(messages); }
    if (settings.enableNsfwClassification) { const images = getImagesToProcess(); if (images.length > 0) processImages(images); }
}
function getCurrentChatName() {
    const headerElement = document.querySelector(".chat-info .person .content .user-title,.chat-info-container .chat-info .title,div.peer-title[data-peer-id]");
    if (headerElement) return (headerElement.querySelector('span.peer-title') || headerElement).textContent.trim();
    const hash = window.location.hash;
    return hash && hash.startsWith('#@') ? hash.substring(2) : null;
}
const observer = new MutationObserver(() => scheduleProcessNewContent());
function init() { loadAndUpdateSettings(); observer.observe(document.body, { childList: true, subtree: true }); log('info', "Black-Fox Content Analyzer Loaded (v2.1)."); setInterval(processNewContent, 3000); }
if (document.readyState === "complete" || document.readyState === "interactive") { init(); } else { document.addEventListener("DOMContentLoaded", init); }