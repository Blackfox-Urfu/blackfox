import os
import json
import csv
import chardet
import sys
from typing import List, Dict, Union

# Увеличиваем лимит для чтения больших CSV-файлов
csv.field_size_limit(sys.maxsize)


def find_all_files(folder_path: str) -> List[str]:
    """Находит все файлы в указанной папке и ее подпапках."""
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def detect_encoding(file_path: str) -> str:
    """Определяет кодировку файла."""
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)
        result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8'


def read_file(file_path: str) -> Union[List[Dict], None]:
    """Читает содержимое файла (JSON, TSV, CSV) и возвращает список словарей."""
    if os.path.getsize(file_path) == 0:
        print(f"Skipping empty file: {file_path}")
        return None

    encoding = detect_encoding(file_path)
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding=encoding) as f:
                content = json.load(f)
                return content if isinstance(content, list) else [content]
        elif file_path.endswith('.tsv'):
            with open(file_path, 'r', encoding=encoding) as f:
                return list(csv.DictReader(f, delimiter='\t'))
        elif file_path.endswith('.csv'):
            with open(file_path, 'r', encoding=encoding) as f:
                return list(csv.DictReader(f))
        else:
            print(f"Unsupported file format: {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return None


def extract_text_fields(data: List[Dict]) -> List[str]:
    """Извлекает текстовые поля (text, comment, message) из данных."""
    texts = []
    for record in data:
        for key in ['text', 'comment', 'message']:
            if key in record and isinstance(record[key], str):
                texts.append(record[key])
    return texts


def process_and_save_files(folder_path: str, output_path: str) -> None:
    """Ищет файлы, парсит их и извлекает текстовые поля с сохранением в файл."""
    all_files = find_all_files(folder_path)

    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text'])  # Заголовок

        for file_path in all_files:
            print(f"Processing file: {file_path}")
            file_data = read_file(file_path)
            if file_data:
                texts = extract_text_fields(file_data)
                for text in texts:
                    writer.writerow([text])
    print(f"Texts successfully saved to {output_path}")


# Путь к папке для обработки
folder_path = "random_dataset_test"

# Путь для сохранения результатов
output_file = "extracted_texts.csv"

# Обработка файлов и сохранение результатов
process_and_save_files(folder_path, output_file)

print("Processing completed.")
