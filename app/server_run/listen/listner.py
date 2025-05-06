from ctypes.util import find_library
from ctypes import *
import json
import os
import sys
import aiohttp
import dotenv
import csv 
import asyncio
import re
from pybloom_live import ScalableBloomFilter
import hashlib
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tdlib_auth.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TDLibAuth')

# Инициализация Bloom Filter
message_filter = ScalableBloomFilter(initial_capacity=1000000, error_rate=0.001)

dotenv.load_dotenv()
api_id = os.getenv('api_id')
api_hash = os.getenv('api_hash')

# Загрузка библиотеки tdjson
tdjson_path = find_library('tdjson')
if tdjson_path is None:
    if os.name == 'nt':
        tdjson_path = os.path.join(os.path.dirname(__file__), 'tdjson.dll')
    else:
        sys.exit("Can't find 'tdjson' library")
tdjson = CDLL(tdjson_path)

# Загрузка функций TDLib
_td_create_client_id = tdjson.td_create_client_id
_td_create_client_id.restype = c_int
_td_create_client_id.argtypes = []

_td_receive = tdjson.td_receive
_td_receive.restype = c_char_p
_td_receive.argtypes = [c_double]

_td_send = tdjson.td_send
_td_send.restype = None
_td_send.argtypes = [c_int, c_char_p]

_td_execute = tdjson.td_execute
_td_execute.restype = c_char_p
_td_execute.argtypes = [c_char_p]

log_message_callback_type = CFUNCTYPE(None, c_int, c_char_p)
_td_set_log_message_callback = tdjson.td_set_log_message_callback
_td_set_log_message_callback.restype = None
_td_set_log_message_callback.argtypes = [c_int, log_message_callback_type]

@log_message_callback_type
def on_log_message_callback(verbosity_level, message):
    if verbosity_level == 0:
        sys.exit('TDLib fatal error: %r' % message)

def td_execute(query):
    query = json.dumps(query).encode('utf-8')
    result = _td_execute(query)
    if result:
        result = json.loads(result.decode('utf-8'))
    return result

_td_set_log_message_callback(2, on_log_message_callback)
td_execute({'@type': 'setLogVerbosityLevel', 'new_verbosity_level': 1})
client_id = _td_create_client_id()

async def classify_message_async(text):
    url = "https://blackfoxus.ru:8000/classify/"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"text": text}) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Ошибка HTTP: {response.status}")
                return None

def is_advertisement(text):
    hashtags = ["#промо", "#партнерский", "#реклама"]
    if any(tag.lower() in text.lower() for tag in hashtags):
        return True
    
    erid_pattern = re.compile(r'(erid|ERID|Erid)\s*:\s*[a-zA-Z0-9]+|ERID\s+[a-zA-Z0-9]+', re.IGNORECASE)
    return bool(erid_pattern.search(text))

def get_message_hash(message):
    return hashlib.sha256(message.strip().encode('utf-8')).hexdigest()

def write_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Message', 'Classification', 'Sender Profile', 'Chat ID'])
        writer.writerow(data)

def start_authorization():
    print("[+] Запуск авторизации...")
    _td_send(client_id, json.dumps({
        '@type': 'setTdlibParameters',
        'database_directory': 'tdlib',
        'use_message_database': True,
        'use_secret_chats': True,
        'api_id': os.getenv('api_id'),
        'api_hash': os.getenv('api_hash'),
        'system_language_code': 'en',
        'device_model': 'Desktop',
        'application_version': '1.0'
    }).encode('utf-8'))
    print("[+] Параметры TDLib отправлены.")


async def main_async():
    logger.info("Starting authorization process")
    start_authorization()
    
    auth_complete = False
    processed_chats = set()
    active_chat_requests = {}  # Для отслеживания текущих запросов по чатам
    message_batch = []         # Для батчинговой записи в CSV
    BATCH_SIZE = 50            # Размер батча для записи в файл

    def write_to_csv(data, filename):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Message', 'Message Hash', 'Classification', 'Sender ID', 'Chat ID', 'Chat Type', 'Chat Title', 'Message Date'])
            writer.writerow(data)

    async def process_chat_history(chat_id, chat_title, chat_type):
        """Оптимизированная загрузка истории сообщений с пагинацией"""
        if chat_id in active_chat_requests:
            return
            
        active_chat_requests[chat_id] = {
            'last_message_id': 0,
            'offset': 0,
            'has_more': True,
            'title': chat_title,
            'type': chat_type
        }
        
        # Первый запрос для чата
        _td_send(client_id, json.dumps({
            '@type': 'getChatHistory',
            'chat_id': chat_id,
            'from_message_id': 0,
            'offset': 0,
            'limit': 100,
            'only_local': False
        }).encode('utf-8'))

    async def process_message_batch():
        """Пакетная обработка накопленных сообщений"""
        if len(message_batch) >= BATCH_SIZE:
            for data in message_batch:
                filename = 'advertisements.csv' if data['is_ad'] else 'regular_messages.csv'
                write_to_csv([
                    data['text'],
                    data['hash'],
                    data['classification'],
                    data['sender_id'],
                    data['chat_id'],
                    data['chat_type'],
                    data['chat_title'],
                    data['date']
                ], filename)
            
            logger.info(f"Processed batch of {len(message_batch)} messages")
            message_batch.clear()

    while True:
        try:
            while True:
                raw_event = _td_receive(1.0)
                if not raw_event:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    # Декодируем байты в строку
                    event_str = raw_event.decode('utf-8')
                    # Преобразуем JSON-строку в словарь
                    event = json.loads(event_str)
                except (UnicodeDecodeError, json.JSONDecodeError) as decode_error:
                    logger.error(f"Failed to decode event: {decode_error}")
                    continue

                if '@type' in event:
                    logger.debug(f"Received event: {event['@type']}")
                else:
                    logger.warning(f"Received event without '@type': {event}")
                
                # Авторизация
                if not auth_complete:
                    if event['@type'] == 'updateAuthorizationState':
                        auth_state = event['authorization_state']['@type']
                        logger.info(f"Authorization state: {auth_state}")
                        
                        if auth_state == 'authorizationStateWaitPhoneNumber':
                            phone_number = input("Введите номер телефона: ")
                            _td_send(client_id, json.dumps({
                                '@type': 'setAuthenticationPhoneNumber',
                                'phone_number': phone_number
                            }).encode('utf-8'))
                            
                        elif auth_state == 'authorizationStateWaitCode':
                            code = input("Введите код из SMS: ")
                            _td_send(client_id, json.dumps({
                                '@type': 'checkAuthenticationCode',
                                'code': code
                            }).encode('utf-8'))
                            
                        elif auth_state == 'authorizationStateWaitPassword':
                            password = input("Введите пароль: ")
                            _td_send(client_id, json.dumps({
                                '@type': 'checkAuthenticationPassword',
                                'password': password
                            }).encode('utf-8'))
                        
                        elif auth_state == 'authorizationStateReady':
                            logger.info("Authorization completed successfully")
                            auth_complete = True
                            
                            # Загружаем список чатов
                            _td_send(client_id, json.dumps({
                                '@type': 'getChats',
                                'limit': 100
                            }).encode('utf-8'))

                # Обработка списка чатов
                elif auth_complete and event['@type'] == 'chats':
                    for chat_id in event['chat_ids']:
                        if chat_id not in processed_chats:
                            processed_chats.add(chat_id)
                            # Получаем информацию о чате
                            _td_send(client_id, json.dumps({
                                '@type': 'getChat',
                                'chat_id': chat_id
                            }).encode('utf-8'))

                # Обработка информации о чате
                elif auth_complete and event['@type'] == 'chat':
                    chat = event
                    chat_id = chat['id']
                    chat_title = chat.get('title', 'Unknown')
                    chat_type = chat['type']['@type']
                    
                    # Обрабатываем только каналы и группы
                    if chat_type in ['chatTypeSupergroup', 'chatTypeBasicGroup', 'chatTypeChannel']:
                        await process_chat_history(chat_id, chat_title, chat_type)

                # Обработка истории сообщений с пагинацией
                elif auth_complete and event['@type'] == 'messages':
                    messages = event['messages']
                    chat_id = messages[0]['chat_id'] if messages else None
                    
                    if chat_id and chat_id in active_chat_requests:
                        chat_data = active_chat_requests[chat_id]
                        
                        for message in messages:
                            if message['content']['@type'] == 'messageText':
                                text = message['content']['text']['text']
                                msg_hash = get_message_hash(text)
                                
                                if msg_hash not in message_filter:
                                    message_filter.add(msg_hash)
                                    
                                    sender = message['sender_id']
                                    sender_id = sender.get('user_id', sender.get('chat_id', 'Unknown'))
                                    date = message['date']
                                    
                                    result = await classify_message_async(text)
                                    if result:
                                        message_batch.append({
                                            'text': text,
                                            'hash': msg_hash,
                                            'classification': result,
                                            'sender_id': sender_id,
                                            'chat_id': chat_id,
                                            'chat_type': chat_data['type'],
                                            'chat_title': chat_data['title'],
                                            'date': date,
                                            'is_ad': is_advertisement(text)
                                        })
                                        
                                        await process_message_batch()
                        
                        # Пагинация: если есть еще сообщения
                        if len(messages) == 100:
                            last_id = min(msg['id'] for msg in messages)
                            active_chat_requests[chat_id]['last_message_id'] = last_id
                            
                            _td_send(client_id, json.dumps({
                                '@type': 'getChatHistory',
                                'chat_id': chat_id,
                                'from_message_id': last_id,
                                'offset': -100,
                                'limit': 100,
                                'only_local': False
                            }).encode('utf-8'))
                        else:
                            # Завершили обработку этого чата
                            logger.info(f"Finished processing {chat_data['title']} (ID: {chat_id})")
                            active_chat_requests.pop(chat_id, None)

                # Обработка новых сообщений в реальном времени
                elif auth_complete and event['@type'] == 'updateNewMessage':
                    message = event['message']
                    if message['content']['@type'] == 'messageText':
                        text = message['content']['text']['text']
                        msg_hash = get_message_hash(text)
                        
                        if msg_hash not in message_filter:
                            message_filter.add(msg_hash)
                            
                            sender = message['sender_id']
                            sender_id = sender.get('user_id', sender.get('chat_id', 'Unknown'))
                            chat_id = message['chat_id']
                            date = message['date']
                            
                            chat_info = td_execute({
                                '@type': 'getChat',
                                'chat_id': chat_id
                            })
                            
                            chat_type = chat_info['type']['@type'] if chat_info else 'unknown'
                            chat_title = chat_info.get('title', 'Unknown')
                            
                            result = await classify_message_async(text)
                            if result:
                                message_batch.append({
                                    'text': text,
                                    'hash': msg_hash,
                                    'classification': result,
                                    'sender_id': sender_id,
                                    'chat_id': chat_id,
                                    'chat_type': chat_type,
                                    'chat_title': chat_title,
                                    'date': date,
                                    'is_ad': is_advertisement(text)
                                })
                                
                                await process_message_batch()

                # Обработка ошибок
                elif event['@type'] == 'error':
                    logger.error(f"TDLib error: {event['message']}")
                    if event['code'] == 429:
                        logger.error("Too many requests, waiting before retry...")
                        await asyncio.sleep(10)
                    elif event['code'] == 404:
                        # Удаляем чат из активных запросов, если он не найден
                        chat_id = next((k for k, v in active_chat_requests.items() 
                                      if v.get('request_id') == event['@extra']), None)
                        if chat_id:
                            logger.warning(f"Chat not found, removing from queue: {chat_id}")
                            active_chat_requests.pop(chat_id, None)

            # Периодическая запись оставшихся сообщений
            if message_batch and len(message_batch) > 0:
                await process_message_batch()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await asyncio.sleep(1)
            
if __name__ == "__main__":
    asyncio.run(main_async())