// Множество для хранения уже обработанных сообщений
const processedMessages = new Set();
const MAX_PROCESSED_MESSAGES = 500; // Ограничение на количество записей

let excludedChannels = [];

// Загружаем настройки фильтрации из хранилища
function updateExcludedChannels() {
    chrome.storage.local.get(['excludedChannels'], (data) => {
        excludedChannels = (data.excludedChannels || []).map(url => url.split('@').pop());
        console.log('Список каналов для обработки обновлен:', excludedChannels);
        updateExcludeButton(); // Обновляем кнопку при изменении списка
    });
}
updateExcludedChannels(); // Загружаем настройки при запуске

function getChannelName() {
    if (typeof document === "undefined") return "";

    const channelElement = document.querySelector(".chat-info .person .content .user-title .peer-title");
    return channelElement ? channelElement.textContent.trim() : "";
}

function updateExcludeButton() {
    const channelName = getChannelName();
    const button = document.getElementById("excludeChannelBtn");

    if (button) {
        if (excludedChannels.includes(channelName)) {
            button.textContent = "(добавить рекламу)";
            button.style.background = "#4CAF50"; // Зеленый цвет
        } else {
            button.textContent = "Уволить админа(исключить рекламу)";
            button.style.background = "#ff5c5c"; // Красный цвет
        }
    }
}

function addExcludeButton() {
    const userTitle = document.querySelector(".chat-info .person .content .user-title");

    if (userTitle && !document.getElementById("excludeChannelBtn")) {
        const button = document.createElement("button");
        button.id = "excludeChannelBtn";
        button.style.marginLeft = "10px";
        button.style.padding = "5px 10px";
        button.style.fontSize = "12px";
        button.style.cursor = "pointer";
        button.style.color = "white";
        button.style.border = "none";
        button.style.borderRadius = "5px";
        button.style.transition = "transform 0.2s ease, background 0.3s ease";

        button.onclick = function () {
            const channelName = getChannelName();

            if (!channelName) return;

            const index = excludedChannels.indexOf(channelName);

            if (index === -1) {
                excludedChannels.push(channelName);
                button.style.background = "#4CAF50"; // Зеленый цвет
                button.textContent = "(добавить рекламу)";
            } else {
                excludedChannels.splice(index, 1);
                button.style.background = "#ff5c5c"; // Красный цвет
                button.textContent = "Уволить админа(исключить рекламу)";
            }

            button.style.transform = "scale(0.9)";
            setTimeout(() => button.style.transform = "scale(1)", 150);

            chrome.storage.local.set({ excludedChannels }, () => {
                console.log(`Канал "${channelName}" ${index === -1 ? "добавлен в" : "удален из"} списка.`);
            });
        };

        userTitle.appendChild(button);
        updateExcludeButton(); // Устанавливаем правильное начальное состояние кнопки
    }
}

function getMessages() {
    const channelName = getChannelName(); // Получаем имя канала каждый раз
    addExcludeButton(); // Добавляем кнопку, если ее нет

    return [...document.querySelectorAll("div.bubble-content > div.message > span.translatable-message")]
        .filter(msg => !processedMessages.has(msg))
        .map(msg => ({
            element: msg,
            text: msg.textContent.trim(),
            channelName
        }));
}

async function classifyMessages(messages) {
    for (const msg of messages) {
        try {
            if (!excludedChannels.includes(msg.channelName)) {
                console.log(`Сообщение из исключенного канала (${msg.channelName}) пропущено.`);
                continue;
            }

            console.log("Отправляем на классификацию:", msg.text);
            const response = await chrome.runtime.sendMessage({
                action: "classify",
                text: msg.text,
            });

            if (!response || typeof response !== "object") {
                console.error("Некорректный ответ от API:", response, "Текст сообщения:", msg.text);
                continue;
            }

            if (response.is_ad) {
                const messageContainer = msg.element.closest("div.bubble-content");
                messageContainer.style.backgroundColor = "#ffcccb";
                messageContainer.style.border = "2px solid #ff0000";
            }

            // Проверка, есть ли уже prediction в элементе
            if (msg.element.querySelector(".prediction")) {
                console.log("Предсказание уже добавлено для этого сообщения.");
                continue;
            }

            const predictionElement = document.createElement("div");
            predictionElement.style.fontSize = "12px";
            predictionElement.style.color = "#888";
            predictionElement.style.marginTop = "5px";
            predictionElement.classList.add("prediction"); // Добавляем класс для удобства поиска
            predictionElement.textContent = `Prediction: ${response.prediction || "Не классифицировано"}`;
            msg.element.appendChild(predictionElement);

            processedMessages.add(msg.element);
        } catch (error) {
            console.error("Ошибка при классификации сообщения:", error, "Текст сообщения:", msg.text);
        }
    }
}


function cleanProcessedMessages() {
    if (processedMessages.size > MAX_PROCESSED_MESSAGES) {
        processedMessages.clear();
        console.log("Очищен список обработанных сообщений");
    }
}

function processNewMessages() {
    updateExcludedChannels(); // Обновляем список исключений перед проверкой
    const messages = getMessages();
    if (messages.length > 0) {
        console.log(`Найдено ${messages.length} новых сообщений.`);
        classifyMessages(messages);
    }
    cleanProcessedMessages(); // Очищаем список обработанных сообщений при необходимости
}

setInterval(processNewMessages, 3000);
console.log("Скрипт content.js загружен и работает.");
