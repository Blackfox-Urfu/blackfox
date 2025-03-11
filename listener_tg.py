from ctypes.util import find_library
from ctypes import *
import json
import os
import sys
import aiohttp
import os
import dotenv
import requests
import csv 
import asyncio

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

# Загрузка функций TDLib из библиотеки
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

# Инициализация логирования TDLib
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

# Установка уровня логирования TDLib
td_execute({'@type': 'setLogVerbosityLevel', 'new_verbosity_level': 1})

# Создание клиента
client_id = _td_create_client_id()

# Асинхронная функция для отправки текста на сервер и получения классификации
async def classify_message_async(text):
    url = "https://blackfoxus.ru:8000/classify/"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"text": text}) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Ошибка HTTP: {response.status}")
                return None

# Функция для записи данных в CSV
def write_to_csv(data):
    file_exists = os.path.isfile('classifications.csv')
    with open('classifications.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Message', 'Classification', 'Sender Profile', 'Chat ID'])
        writer.writerow(data)

# ==============================
# Сразу запускаем авторизацию
# ==============================
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

# Запуск авторизации сразу
start_authorization()

# Основной асинхронный цикл обработки событий
async def main_async():
    while True:
        print("[*] Ожидание события...")
        event = _td_receive(1.0)
        print("[*] Событие получено:", event)
        if event:
            event = json.loads(event.decode('utf-8'))
            print("[*] Обработка события:", event)
            
            # Обработка авторизации
            if event['@type'] == 'updateAuthorizationState':
                auth_state = event['authorization_state']['@type']
                print("[*] Состояние авторизации:", auth_state)

                # Если ждёт номер телефона
                if auth_state == 'authorizationStateWaitPhoneNumber':
                    phone_number = input("Введите ваш номер телефона: ")
                    print("[*] Отправка номера телефона...")
                    _td_send(client_id, json.dumps({
                        '@type': 'setAuthenticationPhoneNumber',
                        'phone_number': phone_number
                    }).encode('utf-8'))
                    print("[*] Номер телефона отправлен.")

                # Ждёт код подтверждения
                elif auth_state == 'authorizationStateWaitCode':
                    code = input("Введите код из SMS: ")
                    print("[*] Отправка кода подтверждения...")
                    _td_send(client_id, json.dumps({
                        '@type': 'checkAuthenticationCode',
                        'code': code
                    }).encode('utf-8'))
                    print("[*] Код подтверждения отправлен.")

                # Ждёт пароль (если включена 2FA)
                elif auth_state == 'authorizationStateWaitPassword':
                    password = input("Введите ваш пароль: ")
                    print("[*] Отправка пароля...")
                    _td_send(client_id, json.dumps({
                        '@type': 'checkAuthenticationPassword',
                        'password': password
                    }).encode('utf-8'))
                    print("[*] Пароль отправлен.")

            # Обработка входящих сообщений
            elif event['@type'] == 'updateNewMessage':
                message_content = event['message']['content']
                if message_content['@type'] == 'messageText':
                    message = message_content['text']['text']
                else:
                    message = "[Не текстовое сообщение]"
                
                # Проверка типа sender_id
                sender_id = event['message']['sender_id']
                if sender_id['@type'] == 'messageSenderUser':
                    sender_profile = sender_id['user_id']
                elif sender_id['@type'] == 'messageSenderChat':
                    sender_profile = sender_id['chat_id']
                else:
                    sender_profile = "[Неизвестный отправитель]"

                chat_id = event['message']['chat_id']
                print(f"[+] Новое сообщение: {message}")
                result = await classify_message_async(message)
                if result:
                    print(f"✅ Классификация: {result}")
                    write_to_csv([message, result, sender_profile, chat_id])

# Запуск асинхронного цикла
asyncio.run(main_async())
