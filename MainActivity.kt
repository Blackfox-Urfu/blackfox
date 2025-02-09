package com.example.telegramwebview

import android.os.Bundle
import android.util.Log
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.webkit.WebChromeClient
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val webView: WebView = findViewById(R.id.webview)
        val webSettings: WebSettings = webView.settings
        
        // Расширенные настройки WebView
        webSettings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            mediaPlaybackRequiresUserGesture = false
            useWideViewPort = true
            loadWithOverviewMode = true
            setSupportZoom(true)
            builtInZoomControls = true
            displayZoomControls = false
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                Log.d(TAG, "Page finished loading: $url")
                injectTelegramExtensionJS(webView)
            }
        }

        webView.webChromeClient = WebChromeClient()
        webView.addJavascriptInterface(TelegramWebInterface(this), "AndroidInterface")
        webView.loadUrl("https://web.telegram.org/")
    }

    private fun injectTelegramExtensionJS(webView: WebView) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // Код из background.js
                val backgroundJs = """
                    window.addEventListener('message', function(event) {
                        if (event.data.action === "classify") {
                            console.log("Запрос на классификацию получен:", event.data.text);

                            fetch("https://blackfoxus.ru:8000/classify/", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ text: event.data.text }),
                            })
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error(`HTTP error! Status: ${response.status}`);
                                }
                                return response.json();
                            })
                            .then(result => {
                                console.log("Ответ от сервера:", result);
                                window.postMessage({ action: "classificationResult", result: result }, "*");
                            })
                            .catch(error => {
                                console.error("Ошибка при отправке ответа:", error);
                                window.postMessage({ action: "classificationResult", result: { is_ad: false, error: error.message || "Unknown error" } }, "*");
                            });
                        }
                    });
                """.trimIndent()

                // Код из content.js
                val contentJs = """
                    const processedMessages = new Set();
                    const MAX_PROCESSED_MESSAGES = 500;
                    let excludedChannels = [];

                    function updateExcludedChannels() {
                        excludedChannels = AndroidInterface.loadExcludedChannels().split(",");
                        console.log('Список каналов для обработки обновлен:', excludedChannels);
                    }
                    updateExcludedChannels();

                    function getChannelName() {
                        const channelElement = document.querySelector(".chat-info .person .content .user-title .peer-title");
                        return channelElement ? channelElement.textContent.trim() : "";
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
                                    button.style.background = "#4CAF50";
                                    button.textContent = "(добавить рекламу)";
                                } else {
                                    excludedChannels.splice(index, 1);
                                    button.style.background = "#ff5c5c";
                                    button.textContent = "Уволить админа(исключить рекламу)";
                                }
                                button.style.transform = "scale(0.9)";
                                setTimeout(() => button.style.transform = "scale(1)", 150);
                                AndroidInterface.saveExcludedChannels(excludedChannels.join(","));
                            };

                            userTitle.appendChild(button);
                        }
                    }

                    function getMessages() {
                        const channelName = getChannelName();
                        addExcludeButton();
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
                                window.postMessage({ action: "classify", text: msg.text }, "*");
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
                        updateExcludedChannels();
                        const messages = getMessages();
                        if (messages.length > 0) {
                            console.log(`Найдено ${messages.length} новых сообщений.`);
                            classifyMessages(messages);
                        }
                        cleanProcessedMessages();
                    }

                    setInterval(processNewMessages, 3000);
                    console.log("Скрипт content.js загружен и работает.");
                """.trimIndent()

                // Объединяем и выполняем JavaScript код
                val combinedJs = "$backgroundJs $contentJs"
                
                withContext(Dispatchers.Main) {
                    webView.evaluateJavascript(combinedJs, null)
                    Log.d(TAG, "JavaScript injected successfully")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error injecting JavaScript", e)
            }
        }
    }
}
