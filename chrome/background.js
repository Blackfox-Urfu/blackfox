chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "classify") {
        console.log("Запрос на классификацию получен:", request.text); // Лог входящего запроса

        fetch("https://blackfoxus.ru:8000/api/classify_text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: request.text }),
        })
        .then(response => {
            console.log("Полчен ответ https:", response);
            if (!response.ok) {
                console.log("!response.ok now")
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(result => {
            console.log("Ответ от сервера:", result); // Лог успешного ответа
            sendResponse(result);
        })
        .catch(error => {
            console.error("Ошибка при отправке ответа:", error);
            sendResponse({ is_ad: false, error: error.message || "Unknown error" });
        });

        return true; // Указывает, что ответ будет отправлен асинхронно
    }
});
