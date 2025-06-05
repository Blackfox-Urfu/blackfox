import json
import os
import glob
from datetime import datetime, timezone

# --- Конфигурация путей ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ИСХОДНЫЕ ПУТИ (не изменяются)
RAW_AD_TELEGRAM_INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "reklama", "result.json")
RAW_NON_AD_TELEGRAM_INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "nereklama", "result.json")
RAW_PIKABU_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "reddit", "text_posts")

# НОВЫЕ ПУТИ ДЛЯ ОБРАБОТАННЫХ ДАННЫХ (для обучения)
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_ADS_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "ads_unified.json")
PROCESSED_NON_ADS_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "non_ads_unified.json")

# --- Вспомогательные функции (остаются без изменений из предыдущей версии) ---

def telegram_text_to_string(text_data):
    """Преобразует поле text из формата Telegram в одну строку."""
    if text_data is None:
        return ""
    if isinstance(text_data, str):
        return text_data.strip()
    if not isinstance(text_data, list):
        return str(text_data).strip()

    parts = []
    for item in text_data:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and 'text' in item:
            content = item.get('text')
            if content is not None:
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.append(telegram_text_to_string(content))
                else:
                    parts.append(str(content))
    return "".join(parts).strip()

def format_timestamp_utc(timestamp_val):
    """Конвертирует Unix timestamp (float или int) в ISO строку и строку Unix timestamp."""
    if timestamp_val is None:
        return "", ""
    try:
        ts = float(timestamp_val)
        dt_object = datetime.fromtimestamp(ts, timezone.utc)
        iso_date = dt_object.isoformat()
        unix_ts_str = str(int(ts))
        return iso_date, unix_ts_str
    except (ValueError, TypeError):
        print(f"Предупреждение: не удалось сконвертировать timestamp: {timestamp_val}")
        return "", ""

def process_telegram_file(filepath, source_filename_override=None):
    """Обрабатывает один Telegram JSON файл и возвращает список унифицированных сообщений."""
    unified_messages = []
    source_base_filename = os.path.basename(filepath) if source_filename_override is None else source_filename_override

    if not os.path.exists(filepath):
        print(f"Предупреждение: Исходный файл не найден {filepath}")
        return unified_messages

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось декодировать JSON из файла {filepath}")
        return unified_messages
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        return unified_messages

    raw_messages = data.get('messages', [])
    if not isinstance(raw_messages, list):
        print(f"Предупреждение: ключ 'messages' отсутствует или не является списком в {filepath}")
        return unified_messages

    for msg in raw_messages:
        if not isinstance(msg, dict):
            print(f"Предупреждение: элемент в 'messages' не является словарем в {filepath}. Пропуск.")
            continue

        text_content = telegram_text_to_string(msg.get('text'))

        sender_name = msg.get('from')
        sender_id = msg.get('from_id')
        if msg.get('type') == 'service' and 'actor' in msg:
            sender_name = msg.get('actor', sender_name)
            sender_id = msg.get('actor_id', sender_id)

        date_iso = msg.get('date', "")
        date_unix_input = msg.get('date_unixtime')
        date_unix_str = str(date_unix_input) if date_unix_input is not None else ""


        if date_unix_str and not date_iso:
            date_iso, _ = format_timestamp_utc(date_unix_str)
        elif date_iso and not date_unix_str:
            try:
                if date_iso.endswith('Z'):
                    dt_obj = datetime.fromisoformat(date_iso[:-1] + '+00:00')
                else:
                    dt_obj = datetime.fromisoformat(date_iso)
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                else:
                    dt_obj = dt_obj.astimezone(timezone.utc)
                date_unix_str = str(int(dt_obj.timestamp()))
            except ValueError as e:
                print(f"Предупреждение: не удалось сконвертировать ISO дату '{date_iso}' в unixtime: {e}")
                date_unix_str = ""

        unified_msg = {
            "id": str(msg.get('id', '')),
            "type": msg.get('type', 'message') or 'message',
            "date": date_iso,
            "date_unixtime": date_unix_str,
            "from": str(sender_name) if sender_name is not None else "",
            "from_id": str(sender_id) if sender_id is not None else "",
            "text": text_content,
            "photo": msg.get('photo', '') or '',
            "file_name": msg.get('file_name', '') or '',
            "text_entities": msg.get('text_entities', []) or [],
            "source_file": source_base_filename,
            "source_type": "telegram"
        }
        if not unified_msg["file_name"] and "file" in msg and isinstance(msg["file"], str):
            if "(File not included" not in msg["file"]:
                 unified_msg["file_name"] = msg["file"]
        unified_messages.append(unified_msg)

    print(f"Обработано {len(unified_messages)} сообщений из {filepath}")
    return unified_messages

def process_pikabu_file(filepath):
    """Обрабатывает один Pikabu JSON файл и возвращает список унифицированных сообщений."""
    unified_messages = []
    source_base_filename = os.path.basename(filepath)

    if not os.path.exists(filepath):
        print(f"Предупреждение: Исходный файл не найден {filepath}")
        return unified_messages

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            posts_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось декодировать JSON из файла {filepath}")
        return unified_messages
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        return unified_messages

    if not isinstance(posts_data, list):
        print(f"Предупреждение: Содержимое файла {filepath} не является списком постов.")
        return unified_messages

    for post in posts_data:
        if not isinstance(post, dict):
            print(f"Предупреждение: Элемент в {filepath} не является словарем (постом). Пропуск.")
            continue

        post_iso_date, post_unix_ts = format_timestamp_utc(post.get('created_utc'))
        post_title = str(post.get('title', '')) if post.get('title') is not None else ''
        post_body_text = str(post.get('text', '')) if post.get('text') is not None else ''
        
        post_text_content = post_title
        if post_body_text:
            post_text_content += ("\n" + post_body_text) if post_title else post_body_text
        post_text_content = post_text_content.strip()
        
        photo_url = ''
        post_url = str(post.get('url', '')) if post.get('url') is not None else ''
        if not post.get('is_self', False) and post_url:
            if any(post_url.lower().endswith(ext) for ext in ('.jpeg', '.jpg', '.png', '.gif')):
                 photo_url = post_url

        unified_post_msg = {
            "id": str(post.get('post_id', '')),
            "type": "message",
            "date": post_iso_date,
            "date_unixtime": post_unix_ts,
            "from": str(post.get('author', '')) if post.get('author') is not None else '',
            "from_id": str(post.get('author', '')) if post.get('author') is not None else '',
            "text": post_text_content,
            "photo": photo_url,
            "file_name": "",
            "text_entities": [],
            "source_file": source_base_filename,
            "source_type": "pikabu_post"
        }
        unified_messages.append(unified_post_msg)

        comments = post.get('comments', [])
        if isinstance(comments, list):
            for comment in comments:
                if not isinstance(comment, dict):
                    continue
                comment_text_content = str(comment.get('text', '')).strip() if comment.get('text') is not None else ''
                comment_iso_date, comment_unix_ts = format_timestamp_utc(comment.get('created_utc'))
                unified_comment_msg = {
                    "id": str(comment.get('id', '')),
                    "type": "message",
                    "date": comment_iso_date,
                    "date_unixtime": comment_unix_ts,
                    "from": str(comment.get('author', '')) if comment.get('author') is not None else '',
                    "from_id": str(comment.get('author', '')) if comment.get('author') is not None else '',
                    "text": comment_text_content,
                    "photo": "",
                    "file_name": "",
                    "text_entities": [],
                    "source_file": source_base_filename,
                    "source_type": "pikabu_comment"
                }
                unified_messages.append(unified_comment_msg)
    print(f"Обработано {len(unified_messages)} сообщений (посты+комментарии) из {filepath}")
    return unified_messages

def save_unified_data(messages, output_filepath, dataset_name_prefix="Processed Data"):
    """Сохраняет список унифицированных сообщений в JSON файл."""
    output_dir = os.path.dirname(output_filepath)
    # Создаем директорию data/processed, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для обработанных данных: {output_dir}")

    timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S")
    root_object = {
        "name": f"{dataset_name_prefix} ({os.path.basename(output_filepath)})",
        "type": "unified_dataset",
        "id": f"merge_run_{timestamp_id}",
        "messages": messages
    }

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(root_object, f, ensure_ascii=False, indent=2)
        print(f"Сохранено {len(messages)} унифицированных сообщений в {output_filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении файла {output_filepath}: {e}")

# --- Основная логика ---
if __name__ == "__main__":
    print("Начало процесса слияния и обработки данных...")

    # Создаем директорию data/processed, если её нет (на всякий случай, save_unified_data тоже это делает)
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Создана директория для обработанных данных: {PROCESSED_DATA_DIR}")

    # 1. Обработка рекламных сообщений
    print("\n--- Обработка рекламных сообщений (из data/raw/reklama) ---")
    ad_messages = process_telegram_file(RAW_AD_TELEGRAM_INPUT_FILE)
    if ad_messages:
        save_unified_data(ad_messages, PROCESSED_ADS_OUTPUT_FILE, "Processed Ads Data")
    else:
        print("Рекламные сообщения не найдены в исходном файле или не обработаны.")

    # 2. Обработка нерекламных сообщений
    print("\n--- Обработка нерекламных сообщений ---")
    all_non_ad_messages = []

    # 2а. Нерекламные из Telegram (из data/raw/nereklama)
    print("Обработка нерекламных Telegram сообщений...")
    non_ad_telegram_messages = process_telegram_file(RAW_NON_AD_TELEGRAM_INPUT_FILE)
    all_non_ad_messages.extend(non_ad_telegram_messages)

    # 2б. Нерекламные из Pikabu (из data/reddit/text_posts)
    print("\nОбработка нерекламных Pikabu сообщений...")
    pikabu_files = glob.glob(os.path.join(RAW_PIKABU_INPUT_DIR, "*.json"))
    if not pikabu_files:
        print(f"Исходные файлы Pikabu не найдены в {RAW_PIKABU_INPUT_DIR}")
    else:
        print(f"Найдено {len(pikabu_files)} Pikabu файлов для обработки.")
    
    for pikabu_file in pikabu_files:
        print(f"Обработка файла: {pikabu_file}")
        non_ad_pikabu_messages = process_pikabu_file(pikabu_file)
        all_non_ad_messages.extend(non_ad_pikabu_messages)
    
    if all_non_ad_messages:
        save_unified_data(all_non_ad_messages, PROCESSED_NON_ADS_OUTPUT_FILE, "Processed Non-Ads Data")
    else:
        print("Нерекламные сообщения не найдены в исходных файлах или не обработаны.")

    print("\nПроцесс слияния и обработки данных завершен.")
    print(f"Обработанные рекламные данные сохранены в: {PROCESSED_ADS_OUTPUT_FILE}")
    print(f"Обработанные нерекламные данные сохранены в: {PROCESSED_NON_ADS_OUTPUT_FILE}")