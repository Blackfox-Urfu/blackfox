package com.example.telegramwebview

import android.os.Bundle
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val webView: WebView = findViewById(R.id.webview)
        val webSettings: WebSettings = webView.settings
        webSettings.javaScriptEnabled = true // Включаем JavaScript

        // Открытие ссылок внутри WebView
        webView.webViewClient = WebViewClient()

        // Подключение интерфейса для взаимодействия с Telegram Web
        webView.addJavascriptInterface(TelegramWebInterface(this), "AndroidInterface")

        // Загружаем Telegram Web
        webView.loadUrl("https://web.telegram.org/")
    }
}
