// Функция для выбора сообщений на странице
function getMessages() {
    // Используем точный селектор для получения всех сообщений
    const messages = document.querySelectorAll("div.bubble-content > div.message > span.translatable-message");
    return Array.from(messages).map(msg => ({
        element: msg,
        text: msg.textContent.trim(),
    }));
}

// Функция для классификации сообщений
async function classifyMessages(messages) {
    for (const msg of messages) {
        try {
            console.log("Отправляем на классификацию:", msg.text); // Лог перед отправкой

            const response = await chrome.runtime.sendMessage({
                action: "classify",
                text: msg.text,
            });

            console.log("Получен ответ:", response); // Лог полученного ответа

            if (!response || typeof response !== "object") {
                console.error("Некорректный ответ от API:", response, "Текст сообщения:", msg.text);
                continue;
            }

            if (response.error) {
                console.error("Ошибка от API:", response.error);
                continue;
            }

            if (response.is_ad) {
                const messageContainer = msg.element.closest("div.bubble-content");
                messageContainer.style.backgroundColor = "#ffcccb";
                messageContainer.style.border = "2px solid #ff0000";
            }

            const predictionText = response.prediction || "Не классифицировано";
            const predictionElement = document.createElement("div");
            predictionElement.style.fontSize = "12px";
            predictionElement.style.color = "#888";
            predictionElement.style.marginTop = "5px";
            predictionElement.textContent = `Prediction: ${predictionText}`;

            msg.element.appendChild(predictionElement);

        } catch (error) {
            console.error("Ошибка при классификации сообщения:", error, "Текст сообщения:", msg.text);
        }
    }
}



// Функция для обновления сообщений
function updateMessages() {
    const messages = getMessages(); // Получаем все сообщения
    classifyMessages(messages); // Отправляем их на классификацию
}

// Запускаем обновление сообщений каждые 3 секунды
setInterval(updateMessages, 3000);

// Лог для проверки загрузки скрипта
console.log("Скрипт content.js загружен и работает.");
