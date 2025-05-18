import os
import hashlib
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

def calculate_file_hash(filepath, hash_algo='md5', chunk_size=4096):
    """
    Вычисляет MD5 или SHA256 хеш файла по частям.
    """
    if hash_algo == 'md5':
        hasher = hashlib.md5()
    elif hash_algo == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError("Unsupported hash algorithm. Choose 'md5' or 'sha256'.")

    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Warning: File not found during hashing: {filepath}")
        return None
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None

def is_image_file(filepath):
    """
    Проверяет, является ли файл изображением, пытаясь его открыть.
    """
    try:
        # Попытка открыть изображение с помощью Pillow
        # Это проверяет, является ли файл корректным изображением, а не просто по расширению
        Image.open(filepath).verify() # verify() не загружает пиксели, только метаданные
        return True
    except (IOError, SyntaxError):
        # Если не удалось открыть или файл поврежден
        return False
    except Exception as e:
        # Другие ошибки (например, недостаточно памяти, хотя для verify это редко)
        print(f"Error checking image file {filepath}: {e}")
        return False

def find_and_delete_duplicate_photos(directory, hash_algo='md5', delete_mode='shortest_path'):
    """
    Находит и удаляет дубликаты фотографий в указанной директории.
    
    Args:
        directory (str): Путь к директории для сканирования.
        hash_algo (str): Алгоритм хеширования ('md5' или 'sha256'). 'md5' быстрее.
        delete_mode (str): Режим выбора файла для сохранения:
                           'first_found': Оставляет первый найденный файл (порядок зависит от os.walk).
                           'shortest_path': Оставляет файл с самым коротким путем (часто оригинальное имя).
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return

    print(f"Scanning directory: {directory}")
    print(f"Using hash algorithm: {hash_algo}")
    print(f"Delete mode: {delete_mode}")
    print("-" * 50)

    # Словарь для хранения хешей: {hash: [filepath1, filepath2, ...]}
    hashes_to_filepaths = defaultdict(list)
    total_files_scanned = 0
    skipped_non_images = 0
    skipped_errors = 0

    # Проход по всем файлам в директории и поддиректориях
    for root, _, files in os.walk(directory):
        for filename in files:
            total_files_scanned += 1
            filepath = os.path.join(root, filename)

            if not os.path.isfile(filepath):
                # Пропускаем не файлы (например, битые симлинки)
                continue

            if not is_image_file(filepath):
                skipped_non_images += 1
                continue

            file_hash = calculate_file_hash(filepath, hash_algo)
            if file_hash:
                hashes_to_filepaths[file_hash].append(filepath)
            else:
                skipped_errors += 1

    print(f"\nScan complete. Scanned {total_files_scanned} files.")
    print(f"Skipped {skipped_non_images} non-image files.")
    print(f"Skipped {skipped_errors} files due to hashing errors.")
    print("-" * 50)

    duplicates_found = 0
    files_to_delete = []

    # Определяем, какие файлы являются дубликатами и должны быть удалены
    for file_hash, filepaths in hashes_to_filepaths.items():
        if len(filepaths) > 1:
            duplicates_found += (len(filepaths) - 1)
            
            if delete_mode == 'first_found':
                keep_file = filepaths[0]
                duplicates_to_delete = filepaths[1:]
            elif delete_mode == 'shortest_path':
                # Сортируем по длине пути (чтобы сохранить "оригинальное" имя, если есть копии)
                # Если длины одинаковые, порядок лексикографический
                sorted_filepaths = sorted(filepaths, key=lambda x: len(x))
                keep_file = sorted_filepaths[0]
                duplicates_to_delete = sorted_filepaths[1:]
            else:
                print(f"Warning: Unknown delete_mode '{delete_mode}'. Skipping duplicates for hash {file_hash}.")
                continue

            print(f"\nFound {len(filepaths)} duplicates for hash: {file_hash}")
            print(f"  Keeping: {keep_file}")
            for dup_file in duplicates_to_delete:
                print(f"  Will delete: {dup_file}")
                files_to_delete.append(dup_file)

    if duplicates_found == 0:
        print("No duplicate photos found.")
        return

    print(f"\nTotal duplicates found: {duplicates_found}")
    print(f"Total files to delete: {len(files_to_delete)}")
    
    if not files_to_delete:
        print("No files marked for deletion after processing.")
        return

    # Запрос подтверждения перед удалением
    confirmation = input("\nDo you want to proceed with deletion? (y/N): ").lower()
    if confirmation != 'y':
        print("Deletion cancelled by user.")
        return

    # Удаление файлов
    deleted_count = 0
    print("\nStarting deletion...")
    for filepath in tqdm(files_to_delete, desc="Deleting files"):
        try:
            os.remove(filepath)
            deleted_count += 1
        except OSError as e:
            print(f"\nError deleting {filepath}: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred while deleting {filepath}: {e}")

    print("-" * 50)
    print(f"Deletion complete. Successfully deleted {deleted_count} files.")
    print(f"Files that could not be deleted: {len(files_to_delete) - deleted_count}")
    print("Consider checking them manually if deletion failed.")

if __name__ == "__main__":
    # Замените на путь к вашей директории с фотографиями
    # Пример:
    # target_directory = "/home/user/Pictures/MyPhotos"
    # target_directory = "C:\\Users\\User\\Desktop\\MyPhotos"
    
    # Можно запросить путь у пользователя
    target_directory = 'data/raw'

    # Выбор режима хеширования и режима удаления
    # hash_algo='md5' - быстрее, 'sha256' - криптографически надежнее, но медленнее
    # delete_mode='first_found' - просто оставляет первый встретившийся дубликат
    # delete_mode='shortest_path' - пытается сохранить файл с самым коротким путем (часто "оригинальное" имя)
    find_and_delete_duplicate_photos(target_directory, hash_algo='sha256', delete_mode='shortest_path')