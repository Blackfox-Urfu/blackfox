private fun injectJavaScript() {
    val jsCode = """
        document.body.style.backgroundColor = "lightblue"; // Изменение фона
        console.log("Content.js работает!");

        // Вызов Android функции
        AndroidInterface.showToast("Расширение работает!");
    """.trimIndent()

    webView.evaluateJavascript(jsCode, null)
}
