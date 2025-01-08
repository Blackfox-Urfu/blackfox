from quart import Quart, websocket, jsonify, request, send_from_directory
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument  # Импортируем типы медиа
import asyncio
import os
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Загрузка конфигурации из .env
load_dotenv()
api_id = os.getenv('TELEGRAM_API_KEY')
api_hash = os.getenv('TELEGRAM_API_HASH')

# Определяем директории
backend_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(backend_DIR)
frontend_DIR = os.path.join(BASE_DIR, "frontend")
logging.info(f"Путь к frontend: {frontend_DIR}")
STATIC_DIR = os.path.join(frontend_DIR, "static")
MEDIA_DIR = os.path.join(STATIC_DIR, "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(backend_DIR, "randfor_best_model.pkl")
VECTORIZER_PATH = os.path.join(backend_DIR, "randfor_vectorizer.pkl")

app = Quart(__name__)

# Создание клиента Telethon
client = TelegramClient('anon', api_id, api_hash)

@app.before_serving
async def startup():
    """Инициализация клиента Telethon перед запуском сервера."""
    logging.info("Запуск TelegramClient...")
    await client.start()
    logging.info("TelegramClient успешно запущен!")

@app.after_serving
async def shutdown():
    """Закрытие клиента Telethon после завершения работы сервера."""
    logging.info("Закрытие TelegramClient...")
    await client.disconnect()
    logging.info("TelegramClient успешно остановлен!")

@app.websocket('/progress')
async def progress():
    """WebSocket для отправки прогресса загрузки и сообщений."""
    try:
        data = await websocket.receive_json()
        logging.info(f"Получена команда от клиента: {data}")
        channel = data.get("channel")
        num_posts = data.get("num_posts")

        if not channel:
            await websocket.send_json({"status": "error", "message": "Channel name is required"})
            return

        if not num_posts or num_posts <= 0:
            await websocket.send_json({"status": "error", "message": "Number of posts must be a positive integer"})
            return

        entity = await client.get_entity(channel)
        messages = await client.get_messages(entity, limit=num_posts)

        if not messages:
            await websocket.send_json({"status": "error", "message": "No messages found for the given channel"})
            return

        total_messages = len(messages)
        result = []

        for idx, msg in enumerate(messages, start=1):
            media_info = None
            if msg.media:
                if isinstance(msg.media, MessageMediaPhoto):
                    downloaded_path = await client.download_media(msg.media, MEDIA_DIR)
                    if downloaded_path:
                        filename = os.path.basename(downloaded_path)
                        media_info = {
                            "type": "photo",
                            "path": f"/media/{filename}"  # Относительный путь для клиента
                        }
                elif isinstance(msg.media, MessageMediaDocument):
                    downloaded_path = await client.download_media(msg.media, MEDIA_DIR)
                    if downloaded_path:
                        filename = os.path.basename(downloaded_path)
                        media_info = {
                            "type": "document",
                            "path": f"/media/{filename}"  # Относительный путь для клиента
                        }

            result.append({
                "id": msg.id,
                "text": msg.message or "",  # Обеспечиваем наличие строки
                "date": msg.date.strftime('%Y-%m-%d %H:%M:%S') if msg.date else None,
                "media": media_info
            })

            # Прогресс обработки
            progress = (idx / total_messages) * 100
            await websocket.send_json({"status": "in_progress", "progress": progress})
            logging.info(f"Сообщение отправлено: {msg.id} Прогресс: {progress:.2f}%")

        # Отправка всех сообщений
        await websocket.send_json({"status": "done", "messages": result})
        logging.info(f"Отправка сообщений клиенту завершена! Всего сообщений: {len(result)}")

    except Exception as e:
        logging.error(f"Ошибка в WebSocket: {e}")
        await websocket.send_json({"status": "error", "message": f"Internal server error: {str(e)}"})
    
    finally:
        # Явное закрытие WebSocket-соединения
        await websocket.close(1000)
        logging.info("WebSocket соединение закрыто.")

@app.route('/media/<path:filename>')
async def media(filename):
    """Маршрут для обслуживания медиафайлов."""
    try:
        logging.info(f"Запрос медиафайла: {filename}")
        return await send_from_directory(MEDIA_DIR, filename)
    except Exception as e:
        logging.error(f"Ошибка при отправке файла {filename}: {e}")
        return jsonify({"error": "File not found"}), 404

@app.route('/fetch_messages', methods=['POST'])
async def fetch_messages():
    """Маршрут для получения сообщений."""
    try:
        data = await request.get_json()
        channel = data.get("channel")
        num_posts = int(data.get("num_posts", 0))

        if not channel:
            return jsonify({"error": "Channel name is required"}), 400

        logging.info(f"Запрос сообщений для канала: {channel}, Количество постов: {num_posts}")
        entity = await client.get_entity(channel)
        messages = await client.get_messages(entity, limit=num_posts)

        result = []
        for msg in messages:
            media_info = None
            if msg.media:
                if isinstance(msg.media, MessageMediaPhoto):
                    downloaded_path = await client.download_media(msg.media, MEDIA_DIR)
                    if downloaded_path:
                        filename = os.path.basename(downloaded_path)
                        media_info = {
                            "type": "photo",
                            "path": f"/media/{filename}"  # Относительный путь для клиента
                        }
                    logging.info(f"Сообщение содержит фотографию: {media_info}")
                elif isinstance(msg.media, MessageMediaDocument):
                    downloaded_path = await client.download_media(msg.media, MEDIA_DIR)
                    if downloaded_path:
                        filename = os.path.basename(downloaded_path)
                        media_info = {
                            "type": "document",
                            "path": f"/media/{filename}"  # Относительный путь для клиента
                        }
                    logging.info(f"Сообщение содержит документ: {media_info}")

            result.append({
                "id": msg.id,
                "text": msg.message or "",  # Обеспечиваем наличие строки
                "date": msg.date.strftime('%Y-%m-%d %H:%M:%S') if msg.date else None,
                "media": media_info,
                "prediction": "Example Prediction"  # Заглушка для предсказания
            })
            logging.info(f"Сообщение обработано: {msg.message}")

        logging.info(f"Сообщений отправлено клиенту: {len(result)}")
        logging.info(f"Возвращаемые данные: {result}")
        return jsonify(result), 200
        logging.info(f"Маршрут /fetch_messages выполнен успешно! result {result}")
    except Exception as e:
        logging.error(f"Ошибка в маршруте /fetch_messages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
async def index():
    """Отправка главной страницы."""
    logging.info(f"Путь к frontend: {frontend_DIR}")
    if os.path.exists(os.path.join(frontend_DIR, 'index.html')):
        logging.info("Файл index.html найден.")
    else:
        logging.error("Файл index.html не найден!")
    return await send_from_directory(frontend_DIR, 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
    logging.info("Сервер запущен!")
