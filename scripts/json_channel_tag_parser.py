import json
import os
import shutil # Для более надежного создания директории, если нужно

def find_tag_in_text_parts(text_parts, tag_to_find):
    """
    Ищет тег в списке текстовых частей сообщения.
    text_parts может быть строкой или списком словарей/строк.
    tag_to_find должен быть в формате '#tag'.
    """
    if isinstance(text_parts, str):
        return tag_to_find in text_parts
    elif isinstance(text_parts, list):
        for part in text_parts:
            if isinstance(part, str):
                if tag_to_find in part:
                    return True
            elif isinstance(part, dict) and "text" in part:
                # Проверяем и "text" внутри словаря, и если сам тип - hashtag
                if part.get("type") == "hashtag" and part.get("text") == tag_to_find:
                    return True
                if isinstance(part["text"], str) and tag_to_find in part["text"]:
                    return True
                # Рекурсивный вызов для вложенных структур, если они ожидаются
                # (в данном примере не нужно, но для общности можно добавить)
                # if isinstance(part["text"], list) and find_tag_in_text_parts(part["text"], tag_to_find):
                #     return True
    return False

def message_contains_tag(message_obj, tag_to_find):
    """
    Проверяет, содержит ли сообщение указанный тег.
    tag_to_find должен быть в формате '#tag'.
    """
    # 1. Проверяем text_entities - это наиболее точный способ найти теги
    if "text_entities" in message_obj and isinstance(message_obj["text_entities"], list):
        for entity in message_obj["text_entities"]:
            if isinstance(entity, dict) and \
               entity.get("type") == "hashtag" and \
               entity.get("text") == tag_to_find:
                return True

    # 2. Проверяем поле "text" (может быть строкой или списком)
    if "text" in message_obj:
        if find_tag_in_text_parts(message_obj["text"], tag_to_find):
            return True
            
    return False

def process_json_file(filepath, tag_to_find, output_dir):
    """
    Обрабатывает один JSON-файл: читает, фильтрует сообщения, сохраняет результат.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось декодировать JSON в файле: {filepath}")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        return

    if not isinstance(data, dict) or "messages" not in data or not isinstance(data["messages"], list):
        print(f"Предупреждение: Файл {filepath} имеет неверную структуру или не содержит ключ 'messages'. Пропуск.")
        return

    filtered_messages = []
    for message in data["messages"]:
        if isinstance(message, dict) and message_contains_tag(message, tag_to_find):
            filtered_messages.append(message)

    if filtered_messages:
        # Создаем новую структуру, копируя метаданные и заменяя сообщения
        output_data = data.copy() # Копируем верхнеуровневые ключи
        output_data["messages"] = filtered_messages

        original_filename = os.path.basename(filepath)
        base, ext = os.path.splitext(original_filename)
        output_filename = f"{base}_filtered_by_{tag_to_find[1:]}{ext}" # Убираем '#' из имени файла
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Отфильтрованный файл сохранен: {output_filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении файла {output_filepath}: {e}")
    else:
        print(f"В файле {filepath} не найдено сообщений с тегом '{tag_to_find}'.")


def main():
    input_dir = input("Укажите путь к директории с JSON файлами: ").strip()
    tag_query = input("Укажите тег для поиска (например, 'нейронки' или '#нейронки'): ").strip()

    if not os.path.isdir(input_dir):
        print(f"Ошибка: Директория '{input_dir}' не найдена.")
        return

    # Форматируем тег: он должен начинаться с '#'
    if not tag_query.startswith('#'):
        tag_to_find = '#' + tag_query
    else:
        tag_to_find = tag_query
        
    print(f"Поиск сообщений с тегом: {tag_to_find}")

    # Создаем выходную директорию
    output_dir_name = os.path.basename(os.path.normpath(input_dir)) + "_filtered"
    # Помещаем выходную директорию рядом с исходной или внутри нее - выберите
    # Вариант 1: Рядом с исходной
    parent_dir = os.path.dirname(os.path.abspath(input_dir))
    output_dir = os.path.join(parent_dir, output_dir_name)
    # Вариант 2: Внутри исходной (менее предпочтительно, чтобы не смешивать)
    # output_dir = os.path.join(input_dir, "filtered_output") 

    os.makedirs(output_dir, exist_ok=True)
    print(f"Результаты будут сохранены в: {output_dir}")

    found_json_files = False
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".json"):
            found_json_files = True
            filepath = os.path.join(input_dir, filename)
            print(f"\nОбработка файла: {filepath}...")
            process_json_file(filepath, tag_to_find, output_dir)
    
    if not found_json_files:
        print(f"В директории {input_dir} не найдено JSON файлов.")

if __name__ == "__main__":
    main()