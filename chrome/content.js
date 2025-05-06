// Множество для хранения уже обработанных сообщений
const processedMessages = new Set();
const MAX_PROCESSED_MESSAGES = 500; // Ограничение на количество записей

let excludedChannels = [];
let displayMode = 'highlight'; // Значение по умолчанию
let threshold = 0.5; // Значение по умолчанию

// Загружаем настройки из хранилища
function updateSettings() {
    chrome.storage.local.get(['excludedChannels', 'displayMode', 'threshold'], (data) => {
        excludedChannels = (data.excludedChannels || []).map(url => url.split('@').pop());
        displayMode = data.displayMode || 'highlight';
        threshold = data.threshold || 0.5;
        console.log('Настройки обновлены:', { excludedChannels, displayMode, threshold });
        updateExcludeButton();
    });
}

// Слушаем изменения в настройках
chrome.storage.onChanged.addListener((changes) => {
    if (changes.excludedChannels) {
        excludedChannels = changes.excludedChannels.newValue.map(url => url.split('@').pop());
    }
    if (changes.displayMode) {
        displayMode = changes.displayMode.newValue;
    }
    if (changes.threshold) {
        threshold = changes.threshold.newValue;
    }
    console.log('Настройки изменены:', { excludedChannels, displayMode, threshold });
});

updateSettings(); // Загружаем настройки при запуске

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

// Применение стилей в зависимости от режима отображения
function applyAdStyle(messageContainer, predictionValue) {
    // Сначала сбрасываем все возможные стили
    messageContainer.style.backgroundColor = '';
    messageContainer.style.border = '';
    messageContainer.style.opacity = '';
    messageContainer.style.display = '';

    if (predictionValue < threshold) {
        return; // Если значение ниже порога, не применяем стили
    }

    switch (displayMode) {
        case 'highlight':
            messageContainer.style.backgroundColor = '#ffcccb';
            messageContainer.style.border = '2px solid #ff0000';
            break;
        case 'hide':
            messageContainer.style.display = 'none';
            break;
        case 'partial':
            messageContainer.style.opacity = '0.5';
            break;
        case 'prediction':
            // Ничего не делаем с контейнером, только показываем предсказание
            break;
    }
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

            const messageContainer = msg.element.closest("div.bubble-content");
            
            // Применяем стили в зависимости от настроек
            if (response.is_ad) {
                applyAdStyle(messageContainer, response.prediction || 0);
            }

            // Добавляем предсказание, если его еще нет
            if (!msg.element.querySelector(".prediction")) {
                const predictionElement = document.createElement("div");
                predictionElement.style.fontSize = "12px";
                predictionElement.style.color = "#888";
                predictionElement.style.marginTop = "5px";
                predictionElement.classList.add("prediction");
                
                // Форматируем значение предсказания
                const predictionValue = response.prediction || 0;
                const formattedPrediction = (predictionValue * 100).toFixed(1) + '%';
                predictionElement.textContent = `Вероятность рекламы: ${formattedPrediction}`;
                
                // Добавляем цветовую индикацию для prediction
                if (predictionValue >= threshold) {
                    predictionElement.style.color = '#ff4444';
                    predictionElement.style.fontWeight = 'bold';
                }

                msg.element.appendChild(predictionElement);
            }

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
    updateSettings(); // Обновляем список исключений перед проверкой
    const messages = getMessages();
    if (messages.length > 0) {
        console.log(`Найдено ${messages.length} новых сообщений.`);
        classifyMessages(messages);
    }
    cleanProcessedMessages(); // Очищаем список обработанных сообщений при необходимости
}

// Обновляем интервал проверки
const checkInterval = setInterval(processNewMessages, 3000);
console.log("Скрипт content.js загружен и работает с новыми настройками.");
