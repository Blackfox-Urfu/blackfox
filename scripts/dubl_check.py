import os
import hashlib
import sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures

# --- Оптимизация №1: Использовать более быстрый алгоритм хеширования ---
# blake2b часто быстрее sha256 при сохранении криптографической стойкости
HASH_ALGO = 'blake2b'
HASHER = getattr(hashlib, HASH_ALGO)

# --- Оптимизация №2: Предварительная фильтрация по расширениям ---
# Пропускаем файлы, которые почти наверняка не являются изображениями
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

def calculate_file_hash(filepath, chunk_size=8192):
    """Вычисляет хеш файла по частям."""
    hasher = HASHER()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (FileNotFoundError, PermissionError):
        # Ошибки доступа лучше обрабатывать в вызывающей функции
        return None
    except Exception:
        return None

def is_image_file(filepath):
    """Проверяет, является ли файл изображением, пытаясь его открыть."""
    try:
        Image.open(filepath).verify()
        return True
    except (IOError, UnidentifiedImageError, SyntaxError):
        return False
    except Exception:
        return False

def process_file(filepath):
    """
    Обрабатывает один файл: проверяет, является ли он изображением, и вычисляет его хеш.
    Возвращает кортеж (хеш, путь_к_файлу) или None.
    """
    # Сначала быстрая проверка по расширению
    if filepath.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    
    # Затем более надежная, но медленная проверка
    if is_image_file(filepath):
        file_hash = calculate_file_hash(filepath)
        if file_hash:
            return file_hash, str(filepath)
    return None

def find_and_handle_duplicate_photos(directory):
    """
    Находит дубликаты фотографий, выводит их список и предлагает пользователю их удалить.
    Сохраняет файл с самым коротким путем.
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        print(f"Ошибка: Директория не найдена: {directory_path}")
        return

    print(f"\nСканирование дубликатов изображений в: {directory_path}")
    hashes_to_filepaths = defaultdict(list)

    # Рекурсивно находим все файлы в директории
    all_files = [p for p in directory_path.rglob('*') if p.is_file()]

    # --- Оптимизация №3: Параллельная обработка файлов ---
    # Используем все доступные ядра процессора для ускорения
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map выполняет process_file для каждого элемента в all_files параллельно
        # tqdm оборачивает для отображения прогресса
        results = list(tqdm(executor.map(process_file, all_files), total=len(all_files), desc="Анализ файлов"))

    # Собираем результаты из параллельных процессов
    for result in results:
        if result:
            file_hash, filepath = result
            hashes_to_filepaths[file_hash].append(filepath)

    # Собираем группы дубликатов и сразу формируем список на удаление
    duplicate_groups = []
    files_to_delete = []
    for filepaths in hashes_to_filepaths.values():
        if len(filepaths) > 1:
            filepaths.sort(key=len)
            duplicate_groups.append(filepaths)
            files_to_delete.extend(filepaths[1:])

    if not files_to_delete:
        print("Дубликаты фотографий не найдены.")
        return

    # --- Вывод списка дубликатов ---
    print(f"\nНайдено {len(files_to_delete)} дубликатов для {len(duplicate_groups)} уникальных изображений.")
    print("-" * 40)
    for group in duplicate_groups:
        print(f"ОРИГИНАЛ (сохраняется): {group[0]}")
        for duplicate in group[1:]:
            print(f"  - ДУБЛИКАТ (к удалению): {duplicate}")
        print()
    print("-" * 40)

    # --- Запрос подтверждения у пользователя ---
    while True:
        try:
            prompt = f"Вы хотите удалить все {len(files_to_delete)} найденных дубликатов? (да/нет): "
            user_choice = input(prompt).lower().strip()
        except EOFError: # Если скрипт запускается неинтерактивно
             print("\nНеинтерактивный режим. Удаление отменено.")
             user_choice = 'нет'

        if user_choice in ["да", "д", "yes", "y"]:
            delete = True
            break
        elif user_choice in ["нет", "н", "no", "n"]:
            delete = False
            break
        else:
            print("Пожалуйста, введите 'да' или 'нет'.")

    if delete:
        print("\nНачинаю удаление...")
        deleted_count = 0
        for filepath in tqdm(files_to_delete, desc="Удаление дубликатов"):
            try:
                os.remove(filepath)
                deleted_count += 1
            except OSError as e:
                print(f"\nОшибка при удалении {filepath}: {e}")
        print(f"Удаление завершено. Успешно удалено {deleted_count} файлов.")
    else:
        print("\nУдаление отменено пользователем. Файлы не были удалены.")

if __name__ == "__main__":
    try:
        PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    except NameError:
        PROJECT_ROOT = Path('.').resolve().parent
        print(f"Переменная __file__ не определена. Установлен корень проекта: {PROJECT_ROOT}")

    target_directory = PROJECT_ROOT / "data"
    find_and_handle_duplicate_photos(target_directory)