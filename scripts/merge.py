import json
import os

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def merge_json_files(reklama_dir, nereklama_dir, reklama_output_file, nereklama_output_file):
    reklama_data = []
    nereklama_data = []
    seen_posts = set()  # Для отслеживания уникальных постов

    # Обработка result.json в папке reklama
    result_filepath = os.path.join(reklama_dir, 'result.json')
    if os.path.isfile(result_filepath):
        result_data = load_json(result_filepath)
        for message in result_data.get('messages', []):
            post_text = extract_text(message)
            if post_text not in seen_posts:
                seen_posts.add(post_text)  # Добавляем посты из result.json в seen_posts
                reklama_data.append(extract_message_data(message))

    # Обработка папки reklama
    for filename in os.listdir(reklama_dir):
        if filename == 'result.json':
            continue  # Пропускаем result.json, так как он уже обработан
        filepath = os.path.join(reklama_dir, filename)
        if os.path.isfile(filepath):
            data = load_json(filepath)
            for message in data.get('messages', []):
                post_text = extract_text(message)
                if post_text not in seen_posts:
                    seen_posts.add(post_text)
                    reklama_data.append(extract_message_data(message))

    # Уникальные посты для nereklama
    seen_posts_nereklama = set(seen_posts)  # Копируем уникальные посты из reklama
    for filename in os.listdir(nereklama_dir):
        filepath = os.path.join(nereklama_dir, filename)
        if os.path.isfile(filepath):
            data = load_json(filepath)
            for message in data.get('messages', []):
                post_text = extract_text(message)
                if post_text not in seen_posts_nereklama:  # Проверяем, есть ли пост в seen_posts
                    seen_posts_nereklama.add(post_text)
                    nereklama_data.append(extract_message_data(message))

    # Сохранение объединенных данных в новые файлы
    with open(reklama_output_file, 'w', encoding='utf-8') as outfile:
        json.dump({'messages': reklama_data}, outfile, ensure_ascii=False, indent=4)

    with open(nereklama_output_file, 'w', encoding='utf-8') as outfile:
        json.dump({'messages': nereklama_data}, outfile, ensure_ascii=False, indent=4)

def extract_text(message):
    full_text = ""
    for part in message["text"]:
        if isinstance(part, dict):
            full_text += part["text"]
        else:
            full_text += part
    return full_text

def extract_message_data(message):
    return {
        "text": extract_text(message),
        "date": message.get("date", ""),
        "from": message.get("from", ""),
        "photo": message.get("photo", ""),
        "file_name": message.get("file_name", "")
    }

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    reklama_dir = os.path.join(BASE_DIR, 'data/reklama')
    nereklama_dir = os.path.join(BASE_DIR, 'data/nereklama')
    reklama_output_file = os.path.join(BASE_DIR, 'data/reklama_data.json')
    nereklama_output_file = os.path.join(BASE_DIR, 'data/nereklama_data.json')

    merge_json_files(reklama_dir, nereklama_dir, reklama_output_file, nereklama_output_file)
    print(f"Merged reklama data saved to {reklama_output_file}")
    print(f"Merged nereklama data saved to {nereklama_output_file}")
