// --- НОВОЕ: Выносим базовый URL в константу ---
const BASE_URL = "https://blackfoxus.ru:8443"; 

const LOGS = [];
const MAX_LOG_ENTRIES = 1000;

function log(level, ...args) {
    const now = new Date().toISOString();
    const message = args.map(arg => (typeof arg === 'object' ? JSON.stringify(arg) : String(arg))).join(' ');
    const logEntry = `${now} [${level.toUpperCase()}] ${message}`;
    LOGS.push(logEntry);
    if (LOGS.length > MAX_LOG_ENTRIES) LOGS.splice(0, LOGS.length - MAX_LOG_ENTRIES);
    console.log(logEntry);
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    log('INFO', `Action: ${request.action} from tab ${sender.tab?.id || 'popup'}`);
    switch (request.action) {
        case "classifyMessage":
            if (request.mode === 'text_only') {
                handleTextOnlyClassification(request, sendResponse);
            } else {
                handleMessageClassification(request, sendResponse);
            }
            return true;
        case "classifyImageNsfw":
            handleNsfwClassification(request, sendResponse);
            return true;
        case "getLogs":
            sendResponse({ logs: LOGS });
            return false;
        case "logFromContent":
            log(request.level || 'INFO', `CONTENT (Tab ${sender.tab?.id}):`, ...request.args);
            return false;
        default:
            log('WARN', `Unknown action: ${request.action}`);
            return false;
    }
});

async function handleMessageClassification(request, sendResponse) {
    log('INFO', `Classifying multimodal. Text: "${(request.text || '').substring(0, 40)}...", Image: ${!!request.imageSrc}`);
    try {
        const formData = new FormData();
        formData.append('text', request.text || '');
        if (request.imageSrc) {
            const blob = await fetchBlob(request.imageSrc);
            if (blob) formData.append('image', blob, `image.${blob.type.split('/')[1] || 'png'}`);
        }
        // --- ИЗМЕНЕНО: Используем константу BASE_URL ---
        const response = await fetch(`${BASE_URL}/api/classify_message/`, { method: "POST", body: formData });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        const result = await response.json();
        log('SUCCESS', `Multimodal result: prob=${result.prediction_prob_ad?.toFixed(4)}`);
        sendResponse(result);
    } catch (error) { log('ERROR', 'Multimodal classification failed:', error.message); sendResponse({ error: error.message }); }
}

async function handleTextOnlyClassification(request, sendResponse) {
    log('INFO', `Classifying text-only. Text: "${(request.text || '').substring(0, 40)}..."`);
    try {
        if (!request.text) throw new Error("Text is required");
        // --- ИЗМЕНЕНО: Используем константу BASE_URL ---
        const response = await fetch(`${BASE_URL}/api/classify_text/`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: request.text }) });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        const result = await response.json();
        log('SUCCESS', `Text-only result: prob=${result.prediction_prob_ad?.toFixed(4)}`);
        sendResponse(result);
    } catch (error) { log('ERROR', 'Text-only classification failed:', error.message); sendResponse({ error: error.message }); }
}

async function handleNsfwClassification(request, sendResponse) {
    log('INFO', `Classifying NSFW. Type: ${request.imageData?.type}`);
    try {
        if (!request.imageData) throw new Error("Missing imageData");
        const blob = request.imageData.type === 'url' ? await fetchBlob(request.imageData.url) : new Blob([new Uint8Array(request.imageData.buffer)], { type: request.imageData.mimeType });
        if (!blob) throw new Error("Failed to get blob");
        const formData = new FormData();
        formData.append('file', blob, `image.${blob.type.split('/')[1] || 'png'}`);
        // --- ИЗМЕНЕНО: Используем константу BASE_URL ---
        const response = await fetch(`${BASE_URL}/api/classify_nsfw_image/`, { method: "POST", body: formData });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        const result = await response.json();
        log('SUCCESS', `NSFW result: prob=${result.prediction_prob_nsfw?.toFixed(4)}`);
        sendResponse(result);
    } catch (error) { log('ERROR', 'NSFW classification failed:', error.message); sendResponse({ error: error.message }); }
}

async function fetchBlob(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.blob();
    } catch (error) { log('ERROR', `Fetch blob failed for URL: ${url}`, error); return null; }
}