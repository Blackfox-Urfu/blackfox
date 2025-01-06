import sys
import asyncio
import logging
import os
import json
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer,QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from dotenv import load_dotenv
import joblib

# ================================
# Конфигурация и инициализация
# ================================

# Настройка логирования
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Загрузка переменных окружения
load_dotenv()
api_id = os.getenv('TELEGRAM_API_KEY')
api_hash = os.getenv('TELEGRAM_API_HASH')

# Загрузка модели и векторизатора
model = joblib.load('randfor_best_model.pkl')
vectorizer = joblib.load('randfor_vectorizer.pkl')

# Создание папки для медиафайлов
MEDIA_DIR = "downloaded_media"
os.makedirs(MEDIA_DIR, exist_ok=True)

# ================================
# Утилитарные функции
# ================================

def predict_ad_content(post):
    """Предсказание вероятности рекламного контента."""
    post_vector = vectorizer.transform([post])
    prediction_proba = model.predict_proba(post_vector)
    prediction = prediction_proba[0][1]
    logging.debug(f"Post: {post[:100]}... Prediction: {prediction}")
    return prediction

# ================================
# Основное приложение
# ================================

class TelegramGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.client = TelegramClient('anon', api_id, api_hash)

    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        self.setWindowTitle("Telegram Content Processor")
        self.setGeometry(100, 100, 800, 600)

        # Основной виджет
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Макет
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Поле для ввода названия канала
        input_layout = QtWidgets.QHBoxLayout()
        self.channel_input = QtWidgets.QLineEdit(self)
        self.channel_input.setPlaceholderText("Enter channel name")
        self.num_posts_input = QtWidgets.QLineEdit(self)
        self.num_posts_input.setPlaceholderText("Number of posts")
        self.start_button = QtWidgets.QPushButton("Fetch Messages", self)
        self.start_button.clicked.connect(self.start_processing)

        # Добавление элементов ввода на макет
        input_layout.addWidget(self.channel_input)
        input_layout.addWidget(self.num_posts_input)
        input_layout.addWidget(self.start_button)
        layout.addLayout(input_layout)

        # Список сообщений
        self.chat_view = QtWidgets.QListWidget(self)
        layout.addWidget(self.chat_view)

    async def fetch_posts(self, channel, num_posts):
        """Асинхронное получение и обработка постов из канала."""
        try:
            async with self.client:
                entity = await self.client.get_entity(channel)
                messages = await self.client.get_messages(entity, limit=num_posts)

                for msg in messages:
                    # Обработка текстовых сообщений
                    if msg.text:
                        content = msg.text
                        prediction = predict_ad_content(content)
                        self.add_message_to_chat(f"Text: {content}\nPrediction: {prediction:.4f}")

                    # Обработка фото
                    if isinstance(msg.media, MessageMediaPhoto):
                        file_path = await msg.download_media(file=os.path.join(MEDIA_DIR, f"photo_{msg.id}.jpg"))
                        self.add_message_to_chat(
                            text="Here's a photo:",
                            media_path=file_path,
                            media_type="image"
                        )

                    # Обработка видео
                    if isinstance(msg.media, MessageMediaDocument) and msg.file.mime_type.startswith("video"):
                        file_path = await msg.download_media(file=os.path.join(MEDIA_DIR, f"video_{msg.id}{os.path.splitext(msg.file.name)[-1]}"))
                        self.add_message_to_chat(
                            text="Here's a video:",
                            media_path=file_path,
                            media_type="video"
                        )
        except Exception as e:
            logging.error(f"Error fetching posts: {e}")
            self.add_message_to_chat(f"Error: {e}")


    def add_message_to_chat(self, text=None, media_path=None, media_type=None):
        """Добавляет сообщение с текстом и медиафайлом в чат."""
        # Основной виджет для сообщения
        message_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(message_widget)
        layout.setSpacing(5)

        # Текст сообщения
        if text:
            text_label = QtWidgets.QLabel(text)
            text_label.setWordWrap(True)
            text_label.setStyleSheet("""
                background-color: #010509;
                border-radius: 10px;
                padding: 8px;
                margin: 2px 10px;
            """)
            layout.addWidget(text_label)

        # Медиа
        if media_path and media_type:
            if media_type == "image":
                # Отображение изображения
                pixmap = QtGui.QPixmap(media_path).scaled(320, 320, QtCore.Qt.KeepAspectRatio)
                media_label = QtWidgets.QLabel()
                media_label.setPixmap(pixmap)
                layout.addWidget(media_label)
            elif media_type == "video":
                # Отображение видео
                video_widget = QVideoWidget()
                layout.addWidget(video_widget)

                media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
                media_player.setVideoOutput(video_widget)
                media_player.setMedia(QMediaContent(QUrl.fromLocalFile(media_path)))
                media_player.play()

        # Добавление сообщения в чат
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(message_widget.sizeHint())
        self.chat_view.addItem(item)
        self.chat_view.setItemWidget(item, message_widget)






    def start_processing(self):
        """Обработчик для кнопки 'Fetch Messages'."""
        channel = self.channel_input.text().strip()
        num_posts = self.num_posts_input.text().strip()

        if not channel or not num_posts.isdigit():
            self.add_message_to_chat("Invalid input. Please enter valid channel name and number of posts.")
            return

        num_posts = int(num_posts)
        asyncio.run(self.fetch_posts(channel, num_posts))

# ================================
# Запуск приложения
# ================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = TelegramGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
