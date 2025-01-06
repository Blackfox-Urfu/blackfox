import asyncio
import time
import logging
import os
import json
from telethon import TelegramClient
import telebot
import joblib
from dotenv import load_dotenv
from threading import Thread

# ================================
# Конфигурация и инициализация
# ================================

# Настройка логирования
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Загрузка переменных окружения
load_dotenv()
api_id = os.getenv('TELEGRAM_API_KEY')
api_hash = os.getenv('TELEGRAM_API_HASH')
bot_token = os.getenv('TELEGRAM_BOT_TOKEN')

# Инициализация Telebot и asyncio-цикла
bot = telebot.TeleBot(bot_token)
loop = asyncio.get_event_loop()

# Загрузка модели и векторизатора
model = joblib.load('randfor_best_model.pkl')
vectorizer = joblib.load('randfor_vectorizer.pkl')

# Хранилище для последних ID сообщений
LAST_MESSAGE_IDS_FILE = "last_message_ids.json"

# Функции для работы с last_message_ids
def save_last_message_ids():
    with open(LAST_MESSAGE_IDS_FILE, "w") as f:
        json.dump(last_message_ids, f)

def load_last_message_ids():
    if os.path.exists(LAST_MESSAGE_IDS_FILE):
        with open(LAST_MESSAGE_IDS_FILE, "r") as f:
            return json.load(f)
    return {}

last_message_ids = load_last_message_ids()

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
# Обработчики команд Telebot
# ================================

@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Отправляет приветственное сообщение и инструкцию по использованию."""
    bot.reply_to(
        message,
        "Напиши:\n1. название канала, откуда выгрузить\n"
        "2. сколько последних постов выгрузить\n"
        "3. куда выгрузить\n\n"
        "Пример ввода:\n `other_channel 10 my_channel`",
    )

@bot.message_handler(content_types=["text"])
def handle_message(message):
    """Обрабатывает текстовые сообщения пользователя."""
    parts = message.text.split()    
    channel, num_posts = parts[:2]
    target_channel = parts[2] if len(parts) > 2 else None
    target_channel = f"@{target_channel}" if target_channel else message.chat.id
    logging.debug(f"Target channel: {target_channel}, type: {type(target_channel)}")
    update = 'update' in parts

    asyncio.run_coroutine_threadsafe(
        handle_message_async(channel, num_posts, target_channel, message, update), loop
    )

# ================================
# Асинхронная обработка сообщений
# ================================

async def handle_message_async(channel, num_posts, target_channel, message, update):
    """Асинхронная обработка сообщений с использованием Telethon."""
    async with TelegramClient('anon', api_id, api_hash) as client:
        try:
            async def fetch_posts():
                """Получение и обработка постов из канала."""
                while True:
                    try:
                        entity = await client.get_entity(channel)
                        limit = 1 if update else int(num_posts)
                        messages = await client.get_messages(entity, limit=limit)

                        if messages:
                            last_message_id = messages[0].id
                            last_handled_id = last_message_ids.get(channel, None)

                            # Если сообщений несколько
                            if last_handled_id and last_message_id > last_handled_id + 1:
                                # Обработка всех пропущенных сообщений
                                new_messages = [
                                    await client.get_messages(entity, ids=msg_id)
                                    for msg_id in range(last_handled_id + 1, last_message_id + 1)
                                ]
                            else:
                                # Если разрывов нет, обработать стандартно
                                new_messages = (
                                    [msg for msg in messages if last_handled_id is None or msg.id > last_handled_id]
                                )

                            for msg in new_messages:
                                await process_message(msg, client, target_channel, channel)

                            if new_messages:
                                last_message_ids[channel] = new_messages[-1].id
                                save_last_message_ids()
                        break

                    except Exception as e:
                        if "429" in str(e):
                            retry_seconds = int(str(e).split("retry after ")[-1].split(".")[0])
                            logging.warning(f"Превышен лимит запросов. Ожидание {retry_seconds} секунд...")
                            await asyncio.sleep(retry_seconds)
                        else:
                            raise e

            if update:
                bot.reply_to(message, "Обновление активировано. Проверка каждые 5 секунд.")
                while update:
                    await fetch_posts()
                    await asyncio.sleep(5)
            else:
                await fetch_posts()
                bot.reply_to(message, "Отлично, пересылка завершена")

        except Exception as e:
            bot.reply_to(
                message,
                f"Ошибка: {e}. Нажми /start, чтобы увидеть правильный формат ввода.",)
            logging.error(f"Async message handling error: {e}")

async def process_message(msg, client, target_channel, channel):
    """Обработка каждого сообщения."""
    caption = msg.text or "No caption provided"  # Используем текст сообщения как подпись (если есть)
    prediction = predict_ad_content(caption)

    try:
        if msg.photo or msg.video:
            # Обработка медиа
            if isinstance(target_channel, int):  # Проверяем, что target_channel - это ID чата
                if msg.photo:
                    sent_msg = bot.send_photo(
                        chat_id=target_channel,
                        photo=msg.photo[-1].file_id,  # Используем последний файл (наивысшее качество)
                        caption=caption
                    )
                elif msg.video:
                    sent_msg = bot.send_video(
                        chat_id=target_channel,
                        video=msg.video.file_id,
                        caption=caption
                    )

                # Отправка результата предсказания
                bot.send_message(
                    chat_id=target_channel,
                    text=f"Prediction: {prediction:.4f}",
                    reply_to_message_id=sent_msg.id
                )
            else:  # Пересылка в канал
                forwarded_msg = await client.forward_messages(
                    target_channel,  # Канал
                    msg.id,  # ID сообщения
                    channel  # Откуда пересылаем
                )
                bot.send_message(
                    chat_id=target_channel,
                    text=f"Prediction: {prediction:.4f}",
                    reply_to_message_id=forwarded_msg.id
                )

        elif msg.message:
            # Обработка текстовых сообщений
            text = msg.message.strip()
            if text:
                prediction = predict_ad_content(text)

                if isinstance(target_channel, int):  # Отправка текста в чат с ботом
                    sent_msg = bot.send_message(chat_id=target_channel, text=text)
                    bot.send_message(
                        chat_id=target_channel,
                        text=f"Prediction: {prediction:.4f}",
                        reply_to_message_id=sent_msg.id
                    )
                else:  # Пересылка текста в канал
                    forwarded_msg = await client.forward_messages(
                        target_channel,
                        msg.id,
                        channel
                    )
                    bot.send_message(
                        chat_id=target_channel,
                        text=f"Prediction: {prediction:.4f}",
                        reply_to_message_id=forwarded_msg.id
                    )
        else:
            logging.warning("Message contains no text, photo, or video. Skipping processing.")

    except Exception as e:
        logging.error(f"Error processing message: {e}")







# ================================
# Основной блок запуска
# ================================

def run_bot():
    """Запуск Telebot в отдельном потоке."""
    bot.polling(none_stop=True, interval=2)

if __name__ == "__main__":
    Thread(target=run_bot).start()

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("Bot stopped.")
