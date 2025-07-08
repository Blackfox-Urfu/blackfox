// Основной обработчик сообщений от content.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "classifyMessage") {
        // Новый обработчик для комплексного анализа сообщений (текст + опционально картинка)
        handleMessageClassification(request, sendResponse);
        return true; // Указываем, что ответ будет асинхронным
    } else if (request.action === "classifyImageNsfw") {
        // Старый обработчик, теперь только для NSFW
        handleNsfwClassification(request, sendResponse);
        return true; // Указываем, что ответ будет асинхронным
    }
});

// НОВАЯ ФУНКЦИЯ: Обработка сообщений на рекламу (текст + картинка)
async function handleMessageClassification(request, sendResponse) {
    try {
        const formData = new FormData();
        // Всегда добавляем текст, даже если он пустой
        formData.append('text', request.text || '');

        // Если есть картинка, получаем её blob и добавляем в форму
        if (request.imageSrc) {
            const blob = await fetchBlob(request.imageSrc);
            if (blob) {
                // Имя файла не критично, но лучше его задать
                formData.append('image', blob, generateFileName(request.imageSrc, blob.type));
            }
        }

        // Отправляем запрос на новый, мультимодальный эндпоинт
        const response = await fetch("http://localhost:8000/api/classify_message/", {
            method: "POST",
            body: formData, // Отправляем FormData, Content-Type установится автоматически
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        sendResponse(result);

    } catch (error) {
        console.error("BG: Ошибка при обработке сообщения:", error);
        sendResponse({
            is_ad: false,
            prediction_prob_ad: 0,
            error: error.message || "Unknown error processing message"
        });
    }
}


// ОБНОВЛЕННАЯ ФУНКЦИЯ: Обработка картинок ТОЛЬКО на NSFW
async function handleNsfwClassification(request, sendResponse) {
    if (!request.imageData) {
        sendResponse({ is_nsfw: false, error: "Missing imageData" });
        return;
    }

    try {
        const blob = await (request.imageData.type === 'url'
            ? fetchBlob(request.imageData.url)
            : Promise.resolve(new Blob([new Uint8Array(request.imageData.buffer)], { type: request.imageData.mimeType }))
        );

        if (!blob) {
            throw new Error("Failed to obtain blob for NSFW classification.");
        }

        const formData = new FormData();
        const fileName = request.imageData.fileName || generateFileName(request.imageData.url, blob.type);
        formData.append('file', blob, fileName);

        // Отправляем запрос на эндпоинт, который специально для NSFW
        const response = await fetch("http://localhost:8000/api/classify_nsfw_image/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        sendResponse(result);

    } catch (error) {
        console.error("BG: Ошибка при обработке NSFW:", error, "Исходные данные:", request.imageData);
        sendResponse({
            is_nsfw: false,
            prediction_prob_nsfw: 0,
            error: error.message || "Unknown error processing NSFW"
        });
    }
}


// Вспомогательные функции (без изменений)
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
        } catch {}
    }
    return `image.${extension}`;
}