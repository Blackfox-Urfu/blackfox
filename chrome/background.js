// Основной обработчик сообщений
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "classify") {
        handleTextClassification(request, sendResponse);
        return true; // Указываем, что ответ будет асинхронным
    } else if (request.action === "classifyAvatar") {
        handleAvatarClassification(request, sendResponse);
        return true; // Указываем, что ответ будет асинхронным
    }
});

// Обработка текстовых сообщений
async function handleTextClassification(request, sendResponse) {
    try {
        const response = await fetch("https://blackfoxus.ru:8000/api/classify_text/ ", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: request.text }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("BG: Ошибка HTTP (текст):", response.status, errorText);
            sendResponse({ 
                is_ad: false, 
                prediction_prob_ad: 0, 
                error: `HTTP error ${response.status}: ${errorText}` 
            });
            return;
        }

        const result = await response.json();
        sendResponse(result);
    } catch (error) {
        console.error("BG: Ошибка при обработке текста:", error);
        sendResponse({ 
            is_ad: false, 
            prediction_prob_ad: 0, 
            error: error.message || "Unknown error processing text" 
        });
    }
}

// Обработка аватарок
async function handleAvatarClassification(request, sendResponse) {
    let blobPromise;
    let finalFileName = 'avatar.png'; // Имя файла по умолчанию

    if (!request.imageData) {
        console.error("BG: Ошибка - imageData отсутствует в запросе на классификацию аватарки.");
        sendResponse({ is_nsfw: false, prediction_prob_nsfw: 0, error: "Missing imageData in request" });
        return;
    }

    if (request.imageData.type === 'arrayBuffer') {
        if (!request.imageData.buffer || !request.imageData.mimeType) {
            console.error("BG: Ошибка - неполные данные для arrayBuffer:", request.imageData);
            sendResponse({ is_nsfw: false, prediction_prob_nsfw: 0, error: "Incomplete ArrayBuffer data" });
            return;
        }
        const byteArray = new Uint8Array(request.imageData.buffer);
        const blob = new Blob([byteArray], { type: request.imageData.mimeType });
        finalFileName = request.imageData.fileName || generateFileName(null, request.imageData.mimeType);
        blobPromise = Promise.resolve(blob);
    } else if (request.imageData.type === 'url') {
        if (!request.imageData.url) {
            console.error("BG: Ошибка - отсутствует URL для типа 'url':", request.imageData);
            sendResponse({ is_nsfw: false, prediction_prob_nsfw: 0, error: "Missing URL for imageData type 'url'" });
            return;
        }
        finalFileName = generateFileName(request.imageData.url);
        blobPromise = fetchBlob(request.imageData.url);
    } else {
        console.error("BG: Неизвестный тип imageData:", request.imageData.type);
        sendResponse({ is_nsfw: false, prediction_prob_nsfw: 0, error: "Unknown imageData type: " + request.imageData.type });
        return;
    }

    try {
        const blob = await blobPromise;
        if (!blob) {
            throw new Error("Failed to obtain blob for avatar.");
        }

        const formData = new FormData();
        formData.append('file', blob, finalFileName);

        const response = await fetch("https://blackfoxus.ru:8000/api/classify_image/ ", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("BG: Ошибка HTTP (аватарка):", response.status, errorText, "Исходные данные:", request.imageData);
            sendResponse({ 
                is_nsfw: false, 
                prediction_prob_nsfw: 0, 
                error: `HTTP error ${response.status}: ${errorText}` 
            });
            return;
        }

        const result = await response.json();
        sendResponse(result);
    } catch (error) {
        console.error("BG: Ошибка при обработке аватарки:", error, "Исходные данные:", request.imageData);
        sendResponse({ 
            is_nsfw: false, 
            prediction_prob_nsfw: 0, 
            error: error.message || "Unknown error processing avatar" 
        });
    }
}

// Вспомогательные функции
async function fetchBlob(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP status ${response.status} for URL: ${url}`);
        return await response.blob();
    } catch (error) {
        console.error("BG: Ошибка получения Blob для URL:", url, error);
        return null;
    }
}

function generateFileName(url, mimeType = 'image/png') {
    let extension = 'png';
    if (mimeType && mimeType.startsWith('image/')) {
        const parts = mimeType.split('/');
        if (parts.length > 1 && ['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(parts[1].toLowerCase())) {
            extension = parts[1].toLowerCase();
        }
    }
    if (url && !url.startsWith('blob:')) {
        try {
            const urlObj = new URL(url);
            const extPart = urlObj.pathname.split('.').pop().toLowerCase();
            if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(extPart)) {
                extension = extPart;
            }
        } catch { /* Игнорируем ошибки парсинга URL */ }
    }
    return `avatar.${extension}`;
}