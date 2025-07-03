import json
import os
import glob
import re
from datetime import datetime, timezone

# --- Конфигурация путей ---
# (Остается без изменений)
# __file__ может не работать в интерактивных средах, но идеально для скриптов
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Запасной вариант для интерактивных сред вроде Jupyter
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RAW_AD_TELEGRAM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "reklama")
RAW_NON_AD_TELEGRAM_INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "nereklama", "result.json")
RAW_PIKABU_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "reddit", "text_posts")

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_ADS_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "ads_unified.json")
PROCESSED_NON_ADS_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "non_ads_unified.json")


# --- Вспомогательные функции ---

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

# --- НОВЫЕ И ОБНОВЛЕННЫЕ ФУНКЦИИ ---

def count_unicode_emojis(text):
    """Считает количество стандартных Unicode эмодзи в строке."""
    # Простой regex для подсчета эмодзи. Может не покрывать абсолютно все, но хорош для начала.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(text))


def extract_telegram_features(msg, text_content):
    """Извлекает числовые и булевы признаки из сообщения."""
    features = {
        "text_length": len(text_content),
        "link_count": 0,
        "mention_count": 0,
        "hashtag_count": 0,
        "bot_command_count": 0,
        "custom_emoji_count": 0,
        "emoji_count": count_unicode_emojis(text_content),
        "has_forward": 'forwarded_from' in msg,
        "has_inline_buttons": bool(msg.get('reply_markup') and msg['reply_markup'].get('rows'))
    }
    
    text_entities = msg.get('text_entities', [])
    if not text_entities:
        return features
        
    for entity in text_entities:
        entity_type = entity.get('type')
        if entity_type in ('url', 'text_link'):
            features['link_count'] += 1
        elif entity_type == 'mention':
            features['mention_count'] += 1
        elif entity_type == 'hashtag':
            features['hashtag_count'] += 1
        elif entity_type == 'bot_command':
            features['bot_command_count'] += 1
        elif entity_type == 'custom_emoji':
            features['custom_emoji_count'] += 1
            
    return features


def extract_telegram_attachments(msg, base_data_dir):
    """Извлекает все медиа-вложения из сообщения в новом формате."""
    attachments = []
    
    # Функция-помощник для проверки существования файла
    def check_file(relative_path):
        if not relative_path or not isinstance(relative_path, str):
            return False, ""
        full_path = os.path.join(base_data_dir, relative_path)
        return os.path.exists(full_path), full_path

    # 1. Фото
    if 'photo' in msg and isinstance(msg['photo'], str):
        is_valid, _ = check_file(msg['photo'])
        attachments.append({
            "type": "photo",
            "path": msg['photo'],
            "is_valid": is_valid,
            "width": msg.get('width'),
            "height": msg.get('height')
        })

    # 2. Видео, GIF, "Кружочки", Голосовые, Стикеры и др. файлы
    media_type = msg.get('media_type')
    # В экспорте Telegram Desktop путь к файлу часто в ключе `file`
    file_path = msg.get('file') or msg.get('file_name')
    
    if not media_type and 'mime_type' in msg:
        mime = msg['mime_type']
        if mime.startswith('video/mp4'):
            media_type = 'video_file'
        elif mime.startswith('audio/'):
            media_type = 'voice_message'
    
    if media_type and file_path:
        attachment = {}
        is_valid, _ = check_file(file_path)
        
        type_map = {
            'video_file': 'video',
            'animation': 'animation', # GIF
            'video_message': 'video_message', # Кружочек
            'voice_message': 'voice_message',
            'sticker': 'sticker',
        }
        attachment['type'] = type_map.get(media_type, 'file') 
        
        attachment['path'] = file_path
        attachment['is_valid'] = is_valid
        attachment['mime_type'] = msg.get('mime_type')
        attachment['duration_seconds'] = msg.get('duration_seconds')
        
        if attachment['type'] in ['video', 'animation', 'video_message']:
            attachment['width'] = msg.get('width')
            attachment['height'] = msg.get('height')
            attachment['has_thumbnail'] = 'thumbnail_path' in msg or 'thumbnail' in msg
        
        if attachment['type'] == 'sticker':
             attachment['sticker_emoji'] = msg.get('sticker_emoji')

        attachments.append(attachment)
        
    return attachments


def process_telegram_file(filepath, source_filename_override=None):
    """Обрабатывает один Telegram JSON файл и возвращает список унифицированных сообщений (НОВАЯ ВЕРСИЯ)."""
    unified_messages = []
    base_data_dir = os.path.dirname(filepath)
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
        if not isinstance(msg, dict) or msg.get('type') == 'service':
            continue

        text_content = telegram_text_to_string(msg.get('text'))
        _, date_unix_str = format_timestamp_utc(msg.get('date_unixtime'))

        unified_msg = {
            "id": str(msg.get('id', '')),
            "type": msg.get('type', 'message'),
            "date_unixtime": date_unix_str,
            "from_id": str(msg.get('from_id')) if msg.get('from_id') else "",
            "source_file": source_base_filename,
            "source_type": "telegram",
            
            "text_content": text_content,
            "text_entities": msg.get('text_entities', []),
            "features": extract_telegram_features(msg, text_content),
            "attachments": extract_telegram_attachments(msg, base_data_dir)
        }
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
            continue

        _, post_unix_ts = format_timestamp_utc(post.get('created_utc'))
        post_title = str(post.get('title', '')).strip()
        post_body_text = str(post.get('text', '')).strip()
        text_content = (f"{post_title}\n{post_body_text}" if post_title and post_body_text else post_title + post_body_text).strip()

        unified_post_msg = {
            "id": str(post.get('post_id', '')),
            "type": "message",
            "date_unixtime": post_unix_ts,
            "from_id": str(post.get('author', '')) if post.get('author') is not None else '',
            "source_file": source_base_filename,
            "source_type": "pikabu_post",
            "text_content": text_content,
            "text_entities": [],
            "features": { "text_length": len(text_content) }, # Добавим базовые фичи
            "attachments": []
        }
        unified_messages.append(unified_post_msg)
        
    print(f"Обработано {len(unified_messages)} сообщений (только посты) из {filepath}")
    return unified_messages


# --- ВОССТАНОВЛЕННАЯ ФУНКЦИЯ ---
def save_unified_data(messages, output_filepath, dataset_name_prefix="Processed Data"):
    """Сохраняет список унифицированных сообщений в JSON файл."""
    output_dir = os.path.dirname(output_filepath)
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

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Создана директория для обработанных данных: {PROCESSED_DATA_DIR}")

    # 1. Обработка рекламных сообщений
    print(f"\n--- Обработка рекламных сообщений (из {RAW_AD_TELEGRAM_DIR}) ---")
    all_ad_messages = []
    ad_json_files = glob.glob(os.path.join(RAW_AD_TELEGRAM_DIR, "*.json"))

    if not ad_json_files:
        print(f"Рекламные JSON файлы не найдены в директории: {RAW_AD_TELEGRAM_DIR}")
    else:
        print(f"Найдено {len(ad_json_files)} рекламных JSON файлов для обработки:")
        for ad_file_path in ad_json_files:
            print(f"  Обработка файла: {ad_file_path}")
            # Вызываем обновленную функцию для Telegram
            messages_from_file = process_telegram_file(ad_file_path)
            all_ad_messages.extend(messages_from_file)
    
    if all_ad_messages:
        save_unified_data(all_ad_messages, PROCESSED_ADS_OUTPUT_FILE, "Processed Ads Data")
    else:
        print("Рекламные сообщения не найдены или не обработаны.")

    # 2. Обработка нерекламных сообщений
    print("\n--- Обработка нерекламных сообщений ---")
    all_non_ad_messages = []

    # 2а. Нерекламные из Telegram
    print(f"Обработка нерекламных Telegram сообщений из {RAW_NON_AD_TELEGRAM_INPUT_FILE}...")
    # Вызываем обновленную функцию для Telegram
    non_ad_telegram_messages = process_telegram_file(RAW_NON_AD_TELEGRAM_INPUT_FILE)
    all_non_ad_messages.extend(non_ad_telegram_messages)

    # 2б. Нерекламные из Pikabu (восстановленная логика)
    print(f"\nОбработка нерекламных Pikabu сообщений из {RAW_PIKABU_INPUT_DIR}...")
    pikabu_files = glob.glob(os.path.join(RAW_PIKABU_INPUT_DIR, "*.json"))
    if not pikabu_files:
        print(f"Исходные файлы Pikabu не найдены в {RAW_PIKABU_INPUT_DIR}")
    else:
        print(f"Найдено {len(pikabu_files)} Pikabu файлов для обработки.")
        for pikabu_file in pikabu_files:
            print(f"  Обработка файла: {pikabu_file}")
            non_ad_pikabu_messages = process_pikabu_file(pikabu_file)
            all_non_ad_messages.extend(non_ad_pikabu_messages)
    
    if all_non_ad_messages:
        save_unified_data(all_non_ad_messages, PROCESSED_NON_ADS_OUTPUT_FILE, "Processed Non-Ads Data")
    else:
        print("Нерекламные сообщения не найдены или не обработаны.")

    print("\nПроцесс слияния и обработки данных завершен.")
    print(f"Обработанные рекламные данные сохранены в: {PROCESSED_ADS_OUTPUT_FILE}")
    print(f"Обработанные нерекламные данные сохранены в: {PROCESSED_NON_ADS_OUTPUT_FILE}")