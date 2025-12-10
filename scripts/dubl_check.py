import os
import hashlib
import sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures

# --- Оптимизация №1: Использовать более быстрый алгоритм хеширования ---
HASH_ALGO = 'blake2b'
HASHER = getattr(hashlib, HASH_ALGO)

# --- Оптимизация №2: Предварительная фильтрация по расширениям ---
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
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        print(f"Ошибка: Директория не найдена: {directory_path}")
        return

    print(f"\nСканирование дубликатов изображений в: {directory_path}")
    hashes_to_filepaths = defaultdict(list)

    # Рекурсивно находим все файлы в директории
    # Преобразуем генератор в список, чтобы знать общее количество для tqdm
    try:
        all_files = [p for p in directory_path.rglob('*') if p.is_file()]
    except PermissionError:
        print("Ошибка: Нет прав доступа к некоторым подпапкам.")
        return

    if not all_files:
        print("В указанной папке файлы не найдены.")
        return

    # --- Оптимизация №3: Параллельная обработка файлов ---
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, all_files), total=len(all_files), desc="Анализ файлов"))

    # Собираем результаты
    for result in results:
        if result:
            file_hash, filepath = result
            hashes_to_filepaths[file_hash].append(filepath)

    # Собираем группы дубликатов
    duplicate_groups = []
    files_to_delete = []
    for filepaths in hashes_to_filepaths.values():
        if len(filepaths) > 1:
            filepaths.sort(key=len) # Оставляем самый короткий путь как оригинал
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

    # --- Запрос подтверждения ---
    while True:
        try:
            prompt = f"Вы хотите удалить все {len(files_to_delete)} найденных дубликатов? (да/нет): "
            user_choice = input(prompt).lower().strip()
        except EOFError:
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

# --- ФУНКЦИЯ ВЫБОРА ПАПКИ ДЛЯ СКАНИРОВАНИЯ ---
def select_scan_directory():
    print("\n--- ВЫБОР ПАПКИ ДЛЯ ПОИСКА ДУБЛИКАТОВ ---")
    
    # Определяем путь по умолчанию (текущая папка проекта/data)
    try:
        default_local = Path(__file__).parent.parent.resolve() / "data"
    except NameError:
        default_local = Path('.').resolve() / "data"

    print(f"1. Локальная папка проекта ({default_local})")
    print("2. Диск sdb1 (ожидается в /mnt/sdb1/reddit_data или корне диска)")
    print("3. Ввести свой путь вручную")
    
    choice = input("Выберите вариант (1-3): ").strip()
    
    target_path = ""
    
    if choice == '1':
        target_path = default_local
    elif choice == '2':
        # Проверяем популярные точки монтирования
        potential_mounts = ["/mnt/sdb1", "/media/sdb1", "/mnt/data", "/media/data"]
        found_mount = None
        
        for mount in potential_mounts:
            if os.path.exists(mount) and os.path.isdir(mount):
                found_mount = mount
                break
        
        if found_mount:
            # Если мы сохраняли загрузчиком в папку reddit_data, ищем её там
            reddit_data_path = os.path.join(found_mount, "reddit_data")
            if os.path.exists(reddit_data_path):
                target_path = reddit_data_path
                print(f"--> Найдена папка с данными: {target_path}")
            else:
                target_path = found_mount
                print(f"--> Папка reddit_data не найдена, сканируем корень диска: {target_path}")
        else:
            print("(!) Не удалось автоматически найти точку монтирования для sdb1.")
            target_path = input("Введите путь к точке монтирования вручную (например /mnt/sdb1): ").strip()
            
    elif choice == '3':
        target_path = input("Введите полный путь к папке для сканирования: ").strip()
    else:
        print("Неверный выбор, используется локальная папка по умолчанию.")
        target_path = default_local

    # Преобразуем в Path и проверяем существование
    path_obj = Path(target_path)
    if not path_obj.exists():
        print(f"\n[ОШИБКА] Указанный путь не существует: {path_obj}")
        sys.exit(1)

    # Проверка прав на запись (нужна для удаления дубликатов)
    try:
        test_file = path_obj / ".write_test_delete_me"
        with open(test_file, 'w') as f: f.write('test')
        os.remove(test_file)
    except PermissionError:
        print(f"\n[ОШИБКА] У вас нет прав на удаление файлов в папке: {path_obj}")
        print("Запустите скрипт через sudo или измените права доступа (chmod).")
        sys.exit(1)
    except Exception as e:
        # Если папка только для чтения, мы не сможем удалить дубликаты
        print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Проблема с правами доступа: {e}")
        
    return path_obj

if __name__ == "__main__":
    # Выбираем папку
    target_directory = select_scan_directory()
    
    # Запускаем поиск
    find_and_handle_duplicate_photos(target_directory)