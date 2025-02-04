browser.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "classify") {
        console.log("Запрос на классификацию получен:", request.text);

        fetch("http://127.0.0.1:8000/classify/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: request.text }),
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(result => {
                console.log("Ответ от сервера:", result);
                sendResponse(result);
            })
            .catch(error => {
                console.error("Ошибка при отправке ответа:", error);
                sendResponse({ is_ad: false, error: error.message || "Unknown error" });
            });

        return true; // Указывает, что ответ будет отправлен асинхронно
    }
});
