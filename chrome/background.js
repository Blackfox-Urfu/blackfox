// --- ОБНОВЛЕНО: Порт 8443 убран, Nginx принимает запросы на стандартном 443 ---
const BASE_URL = "https://blackfoxus.ru"; 

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
    // Упростил лог, чтобы не спамить ID вкладки, если его нет
    const origin = sender.tab ? `tab ${sender.tab.id}` : 'popup';
    log('INFO', `Action: ${request.action} from ${origin}`);

    switch (request.action) {
        case "classifyMessage":
            if (request.mode === 'text_only') {
                handleTextOnlyClassification(request, sendResponse);
            } else {
                handleMessageClassification(request, sendResponse);
            }
            // Важно: возвращаем true, чтобы указать, что ответ будет отправлен асинхронно
            return true;
        case "classifyImageNsfw":
            handleNsfwClassification(request, sendResponse);
            return true;
        case "getLogs":
            sendResponse({ logs: LOGS });
            return false;
        case "logFromContent":
            log(request.level || 'INFO', `CONTENT (${origin}):`, ...request.args);
            return false;
        default:
            log('WARN', `Unknown action: ${request.action}`);
            return false;
    }
});

async function handleMessageClassification(request, sendResponse) {
    log('INFO', `Classifying multimodal. Text len: ${request.text ? request.text.length : 0}, Image: ${!!request.imageSrc}`);
    try {
        const formData = new FormData();
        formData.append('text', request.text || '');
        
        if (request.imageSrc) {
            const blob = await fetchBlob(request.imageSrc);
            if (blob) {
                // Генерируем имя файла, так как сервер может валидировать расширение
                const ext = blob.type.split('/')[1] || 'png';
                formData.append('image', blob, `image.${ext}`);
            }
        }

        const response = await fetch(`${BASE_URL}/api/classify_message/`, { 
            method: "POST", 
            body: formData 
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        
        const result = await response.json();
        log('SUCCESS', `Multimodal result: is_ad=${result.is_ad}, prob=${result.prediction_prob_ad?.toFixed(4)}`);
        sendResponse(result);
    } catch (error) { 
        log('ERROR', 'Multimodal classification failed:', error.message); 
        sendResponse({ error: error.message }); 
    }
}

async function handleTextOnlyClassification(request, sendResponse) {
    log('INFO', `Classifying text-only. Text len: ${request.text ? request.text.length : 0}`);
    try {
        if (!request.text) throw new Error("Text is required");
        
        const response = await fetch(`${BASE_URL}/api/classify_text/`, { 
            method: "POST", 
            headers: { "Content-Type": "application/json" }, 
            body: JSON.stringify({ text: request.text }) 
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        
        const result = await response.json();
        log('SUCCESS', `Text-only result: is_ad=${result.is_ad}, prob=${result.prediction_prob_ad?.toFixed(4)}`);
        sendResponse(result);
    } catch (error) { 
        log('ERROR', 'Text-only classification failed:', error.message); 
        sendResponse({ error: error.message }); 
    }
}

async function handleNsfwClassification(request, sendResponse) {
    log('INFO', `Classifying NSFW. Data type: ${request.imageData?.type}`);
    try {
        if (!request.imageData) throw new Error("Missing imageData");
        
        let blob;
        if (request.imageData.type === 'url') {
            blob = await fetchBlob(request.imageData.url);
        } else if (request.imageData.buffer) {
            // Преобразуем сериализованный буфер обратно в Blob
            // (так как Chrome не позволяет передавать Blob напрямую через сообщения)
            blob = new Blob([new Uint8Array(request.imageData.buffer)], { type: request.imageData.mimeType });
        }

        if (!blob) throw new Error("Failed to create image blob");

        const formData = new FormData();
        const ext = blob.type.split('/')[1] || 'png';
        formData.append('file', blob, `image.${ext}`);

        const response = await fetch(`${BASE_URL}/api/classify_nsfw_image/`, { 
            method: "POST", 
            body: formData 
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        
        const result = await response.json();
        log('SUCCESS', `NSFW result: is_nsfw=${result.is_nsfw}, prob=${result.prediction_prob_nsfw?.toFixed(4)}`);
        sendResponse(result);
    } catch (error) { 
        log('ERROR', 'NSFW classification failed:', error.message); 
        sendResponse({ error: error.message }); 
    }
}

async function fetchBlob(url) {
    try {
        // Важно: для работы fetch с внешними URL в манифесте расширения
        // должны быть прописаны разрешения (host_permissions)
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.blob();
    } catch (error) { 
        log('ERROR', `Fetch blob failed for URL: ${url}`, error.message); 
        return null; 
    }
}