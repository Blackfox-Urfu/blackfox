chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
    if (request.action === "classify") {
        try {
            const response = await fetch("http://127.0.0.1:8000/classify/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: request.text }),
            });
            const result = await response.json();
            sendResponse(result);
        } catch (error) {
            console.error("Error classifying message:", error);
            sendResponse({ is_ad: false });
        }
    }
    return true; // Указывает на асинхронный ответ
});
