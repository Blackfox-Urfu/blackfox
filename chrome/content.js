// Множество для хранения уже обработанных сообщений
const processedMessages = new Set();

// Функция для выбора сообщений на странице
function getMessages() {
    // Используем точный селектор для получения всех сообщений
    const messages = document.querySelectorAll("div.bubble-content > div.message > span.translatable-message");
    return Array.from(messages)
        .filter(msg => !processedMessages.has(msg)) // Фильтруем только необработанные сообщения
        .map(msg => ({
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

            console.log(`[${new Date().toISOString()}] Получен ответ:`, response);

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

            // Добавляем сообщение в множество обработанных
            processedMessages.add(msg.element);

        } catch (error) {
            console.error("Ошибка при классификации сообщения:", error, "Текст сообщения:", msg.text);
        }
    }
}

// Функция для обработки новых сообщений
function processNewMessages() {
    const messages = getMessages(); // Получаем только необработанные сообщения
    if (messages.length > 0) {
        classifyMessages(messages);
    }
}

// Запускаем проверку новых сообщений раз в 3 секунды
setInterval(processNewMessages, 3000);

console.log("Скрипт content.js загружен и работает.");
