import asyncio
import time
from telethon import TelegramClient
import telebot
import logging
import joblib
from dotenv import load_dotenv
import os

# Настройка логирования
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Загрузка модели и векторизатора
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Загрузка переменных окружения
load_dotenv()
api_id = os.getenv('TELEGRAM_API_KEY')
api_hash = os.getenv('TELEGRAM_API_HASH')
bot_token = os.getenv('TELEGRAM_BOT_TOKEN')

# Инициализация Telebot
bot = telebot.TeleBot(bot_token)

# Инициализация глобального asyncio-цикла
loop = asyncio.get_event_loop()

# Словарь для хранения последнего ID сообщений
last_message_ids = {}


def predict_ad_content(post):
    post_vector = vectorizer.transform([post])
    prediction_proba = model.predict_proba(post_vector)
    prediction = prediction_proba[0][1]  # Вероятность класса с меткой 1 (реклама)
    logging.debug(f"Post: {post[:100]}... Prediction: {prediction}")
    return prediction  # Возвращает вероятность класса "реклама" от 0 до 1


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(
        message,
        "Напиши:\n1. название канала, откуда выгрузить\n2. сколько последних постов выгрузить\n3. куда выгрузить\n\nПример ввода:\n `other_channel 10 my_channel`",
    )


@bot.message_handler(content_types=["text"])
def handle_message(message):
    parts = str(message.text).split()

    if len(parts) < 3:
        bot.reply_to(
            message, "Неправильный формат. Нажми /start, чтобы увидеть правильный формат ввода."
        )
        return

    channel, num_posts, target_channel = parts[:3]
    target_channel = "@" + target_channel
    update = 'update' in parts

    # Передаем обработку в асинхронный контекст
    asyncio.run_coroutine_threadsafe(
        handle_message_async(channel, num_posts, target_channel, message, update),
        loop
    )


async def handle_message_async(channel, num_posts, target_channel, message, update):
    async with TelegramClient('anon', api_id, api_hash) as client:
        try:
            async def fetch_posts():
                while True:  # Добавлен цикл для повторной попытки в случае ошибки 429
                    try:
                        entity = await client.get_entity(channel)
                        messages = await client.get_messages(entity, limit=int(num_posts))

                        if messages:
                            last_message_id = messages[0].id
                            last_handled_id = last_message_ids.get(channel, None)

                            # Фильтрация только новых сообщений
                            new_messages = [
                                msg
                                for msg in messages
                                if msg.id > last_handled_id
                            ] if last_handled_id else messages

                            for msg in new_messages[::-1]:
                                if msg.photo or msg.video:
                                    caption = msg.text
                                    if caption:
                                        prediction = predict_ad_content(caption)
                                        if prediction:
                                            forwarded_msg = await client.forward_messages(target_channel, msg.id, channel)
                                            # Отправка предсказания как reply
                                            bot.send_message(
                                                target_channel,
                                                f"Prediction: {prediction:.4f}",
                                                reply_to_message_id=forwarded_msg.id
                                            )
                                            last_message_ids[channel] = msg.id  # Обновляем последний ID

                                elif msg.message:
                                    text = msg.message
                                    if text and text.strip():
                                        prediction = predict_ad_content(text)
                                        if prediction <= 0.7:
                                            sent_msg = bot.send_message(target_channel, text)
                                            # Отправка предсказания как reply
                                            bot.send_message(
                                                target_channel,
                                                f"Prediction: {prediction:.4f}",
                                                reply_to_message_id=sent_msg.message_id
                                            )
                                            last_message_ids[channel] = msg.id  # Обновляем последний ID
                        break  # Если запрос успешный, выходим из цикла

                    except Exception as e:
                        if "429" in str(e):
                            retry_seconds = int(
                                str(e).split("retry after ")[-1].split(".")[0]
                            )
                            logging.warning(
                                f"Превышен лимит запросов. Ожидание {retry_seconds} секунд..."
                            )
                            time.sleep(retry_seconds)  # Ждем перед повторной попыткой
                        else:
                            raise e  # Если ошибка не 429, выбрасываем дальше


            if update:
                bot.reply_to(message, "Обновление активировано. Ждемc...")
                await fetch_posts()
            else:
                await fetch_posts()
                bot.reply_to(message, "Отлично, пересылка завершена")

        except Exception as e:
            bot.reply_to(
                message,
                f"Ошибка: {e}. Нажми /start, чтобы увидеть правильный формат ввода.",
            )
            logging.error(f"Async message handling error: {e}")


if __name__ == "__main__":
    # Запуск Telebot в отдельном потоке
    from threading import Thread

    def run_bot():
        bot.polling(none_stop=True, interval=2)

    # Запуск asyncio-цикла в основном потоке
    Thread(target=run_bot).start()

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("Bot stopped.")
