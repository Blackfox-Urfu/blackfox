// Выбираем сообщения на странице
function getMessages() {
    // Используем точный селектор
    const messages = document.querySelectorAll("div.bubble-content > div.message > span.translatable-message");
    return Array.from(messages).map(msg => ({
        element: msg,
        text: msg.textContent.trim(),
    }));
}

// Отправляем сообщения для классификации
async function classifyMessages(messages) {
    for (const msg of messages) {
        try {
            const response = await chrome.runtime.sendMessage({
                action: "classify",
                text: msg.text,
            });
            
            // Проверяем наличие ошибки в ответе
            if (response.error) {
                console.error("Ошибка от API:", response.error);
                continue;
            }
            // Если сообщение реклама, выделяем его
            if (response.is_ad) {
                const messageContainer = msg.element.closest("div.bubble-content");

                // Выделяем рекламное сообщение (например, меняем фон)
                messageContainer.style.backgroundColor = "#ffcccb"; // Красный фон для выделения
                messageContainer.style.border = "2px solid #ff0000"; // Красная рамка
            }

            // Добавляем значение prediction в сообщение
            const predictionText = response.prediction || "Не классифицировано";
            const predictionElement = document.createElement("div");
            predictionElement.style.fontSize = "12px";
            predictionElement.style.color = "#888";  // Серый цвет
            predictionElement.style.marginTop = "5px";
            predictionElement.textContent = `Prediction: ${predictionText}`;

            // Добавляем блок с prediction после текста сообщения
            msg.element.appendChild(predictionElement);

        } catch (error) {
            console.error("Ошибка при классификации сообщения:", error);
        }
    }
}

// Обновляем фильтрацию при загрузке страницы
setInterval(() => {
    const messages = getMessages();
    classifyMessages(messages);
}, 3000);
