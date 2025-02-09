package com.example.telegramwebview

import android.content.Context
import android.webkit.JavascriptInterface
import android.widget.Toast

class TelegramWebInterface(private val context: Context) {

    /**
     * Метод, который можно вызвать из JavaScript в WebView.
     * Например, Telegram Web может передавать сюда сообщения.
     */
    @JavascriptInterface
    fun showToast(message: String) {
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }

    /**
     * Метод для обработки сообщений от Telegram Web.
     * Можно модифицировать для работы с логикой приложения.
     */
    @JavascriptInterface
    fun onMessageReceived(message: String) {
        // Здесь можно обработать полученное сообщение (например, отправить в лог)
        println("Сообщение из WebView: $message")
    }
}
