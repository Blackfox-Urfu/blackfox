import android.content.Context
import android.webkit.JavascriptInterface
import android.widget.Toast

class TelegramWebInterface(private val context: Context) {
    private val sharedPreferences = context.getSharedPreferences("TelegramPrefs", Context.MODE_PRIVATE)

    @JavascriptInterface
    fun sendMessage(message: String) {
        // Реализация отправки сообщений
    }

    @JavascriptInterface
    fun getContacts() {
        // Реализация получения контактов
    }

    @JavascriptInterface
    fun loadExcludedChannels(): String {
        val channels = sharedPreferences.getStringSet("excludedChannels", emptySet()) ?: emptySet()
        return channels.joinToString(",")
    }

    @JavascriptInterface
    fun saveExcludedChannels(channels: String) {
        val channelSet = channels.split(",").toSet()
        sharedPreferences.edit().putStringSet("excludedChannels", channelSet).apply()
        Toast.makeText(context, "Список каналов обновлен", Toast.LENGTH_SHORT).show()
    }

    @JavascriptInterface
    fun classifyMessage(text: String) {
        // Здесь можно реализовать логику классификации на стороне Android, если необходимо
        Toast.makeText(context, "Классификация сообщения: $text", Toast.LENGTH_SHORT).show()
    }

    // Добавьте другие методы, которые соответствуют функционалу расширения Chrome
} 