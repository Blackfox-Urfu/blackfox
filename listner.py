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
    
    while True:
        try:
            event = _td_receive(1.0)
            if event:
                event = json.loads(event.decode('utf-8'))
                logger.debug(f"Received event: {json.dumps(event, indent=2)}")
                
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
                            # Загружаем чаты после авторизации
                            _td_send(client_id, json.dumps({
                                '@type': 'loadChats',
                                'limit': 100
                            }).encode('utf-8'))
                
                # Обработка сообщений после авторизации
                if auth_complete and event['@type'] == 'updateNewMessage':
                    if event['message']['content']['@type'] == 'messageText':
                        message = event['message']['content']['text']['text']
                        msg_hash = get_message_hash(message)
                        
                        if msg_hash not in message_filter:
                            message_filter.add(msg_hash)
                            
                            sender = event['message']['sender_id']
                            sender_id = sender.get('user_id', sender.get('chat_id', 'Unknown'))
                            chat_id = event['message']['chat_id']
                            
                            result = await classify_message_async(message)
                            if result:
                                filename = 'advertisements.csv' if is_advertisement(message) else 'regular_messages.csv'
                                write_to_csv([message, result, sender_id, chat_id], filename)
                                logger.info(f"Processed message from chat {chat_id}")
                
                elif event['@type'] == 'error':
                    logger.error(f"TDLib error: {event['message']}")
                    if event['code'] == 429:
                        logger.error("Too many requests, waiting before retry...")
                        await asyncio.sleep(10)
                
                elif event['@type'] == 'updateConnectionState':
                    logger.info(f"Connection state: {event['state']['@type']}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await asyncio.sleep(1)
            
if __name__ == "__main__":
    asyncio.run(main_async())