import asyncio
from telethon import TelegramClient
import telebot
import logging
from keys import api_id, api_hash, bot_token
import joblib
import time

# Настройка логирования
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Загрузка модели и векторизатора
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Инициализация клиента Telethon
client = TelegramClient('anon', api_id, api_hash)

# Инициализация бота Telebot
bot = telebot.TeleBot(bot_token)

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
    bot.reply_to(message, "Напиши:\n1. название канала, откуда выгрузить\n2. сколько последних постов выгрузить\n3. куда выгрузить\n\nПример ввода:\n `other_channel 10 my_channel`")

@bot.message_handler(content_types=["text"])
def handle_message(message):
    asyncio.run(handle_message_async(message))

async def handle_message_async(message):
    try:
        parts = str(message.text).split()

        if len(parts) < 3:
            bot.reply_to(message, "Неправильный формат. Нажми /start, чтобы увидеть правильный формат ввода.")
            return
        
        channel, num_posts, target_channel = parts[:3]
        target_channel = "@" + target_channel
        update = 'update' in parts

        async def fetch_posts():
            async with client:
                entity = await client.get_entity(channel)
                messages = await client.get_messages(entity, limit=int(num_posts))

                if messages:
                    last_message_id = messages[0].id
                    last_handled_id = last_message_ids.get(channel, None)

                    # Фильтрация только новых сообщений
                    new_messages = [msg for msg in messages if msg.id > last_handled_id] if last_handled_id else messages

                    for msg in new_messages[::-1]:
                        if msg.photo or msg.video:
                            caption = msg.text
                            if caption:
                                prediction = predict_ad_content(caption)
                                if prediction:
                                    await client.forward_messages(target_channel, msg.id, channel)
                                    bot.send_message(target_channel, f"Prediction: {prediction:.4f}")
                                    last_message_ids[channel] = msg.id  # Обновляем последний ID после успешной пересылки

                        elif msg.message:
                            text = msg.message
                            if text and text.strip():
                                prediction = predict_ad_content(text)
                                if prediction <= 0.7:
                                    bot.send_message(target_channel, text)
                                    bot.send_message(target_channel, f"Prediction: {prediction:.4f}")
                                    last_message_ids[channel] = msg.id  # Обновляем последний ID после успешной пересылки

        async def periodic_fetch():
            retry_after = 1  # Изначальный интервал между запросами
            total_forwarded = 0  # Счётчик отправленных сообщений

            while True:
                try:
                    await fetch_posts()
                    total_forwarded += len(last_message_ids)  # Увеличиваем счётчик на количество новых сообщений
                    logging.debug(f"Всего переслано сообщений: {total_forwarded}")
                    await asyncio.sleep(retry_after)
                except Exception as e:
                    if "429" in str(e):
                        retry_seconds = int(str(e).split("retry after ")[-1].split(".")[0])
                        logging.error(f"Превышен лимит запросов. Пересылка приостановлена на {retry_seconds} секунд.")
                        bot.send_message(message.chat.id, f"Превышен лимит запросов. Пересылка приостановлена до {time.strftime('%H:%M:%S', time.localtime(time.time() + retry_seconds))}.")
                        await asyncio.sleep(retry_seconds)
                        bot.send_message(message.chat.id, "Пересылка возобновлена.")
                    else:
                        raise e

        if update:
            bot.reply_to(message, "Обновление активировано. Пожалуйста, подождите...")
            await periodic_fetch()
        else:
            await fetch_posts()
            bot.reply_to(message, "Отлично, пересылка завершена")

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}. Нажми /start, чтобы увидеть правильный формат ввода.")
        logging.error(f"Message handling error: {e}")

if __name__ == "__main__":
    while True:
        try:
            bot.polling(none_stop=True, interval=2)
        except Exception as e:
            logging.error(f"Polling error: {e}. Reconnecting...")
            time.sleep(5)
