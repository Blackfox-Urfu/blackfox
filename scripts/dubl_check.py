import os
import hashlib
import sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from collections import defaultdict

def calculate_file_hash(filepath, hash_algo='sha256', chunk_size=8192):
    """Вычисляет SHA256 хеш файла по частям."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (FileNotFoundError, PermissionError) as e:
        print(f"Warning: Could not access file for hashing: {filepath} ({e})")
        return None
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None

def is_image_file(filepath):
    """Проверяет, является ли файл изображением, пытаясь его открыть."""
    try:
        Image.open(filepath).verify()
        return True
    except (IOError, UnidentifiedImageError, SyntaxError):
        return False
    except Exception as e:
        print(f"Error checking image file {filepath}: {e}")
        return False

def find_and_delete_duplicate_photos(directory, delete=False):
    """
    Находит и опционально удаляет дубликаты фотографий в указанной директории.
    Сохраняет файл с самым коротким путем.
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        print(f"Error: Directory not found: {directory_path}")
        return

    print(f"\nScanning for duplicate images in: {directory_path}")
    hashes_to_filepaths = defaultdict(list)

    # Рекурсивно находим все файлы в директории
    all_files = [p for p in directory_path.rglob('*') if p.is_file()]

    for filepath in tqdm(all_files, desc=f"Hashing images in {directory_path.name}"):
        if is_image_file(filepath):
            file_hash = calculate_file_hash(filepath)
            if file_hash:
                hashes_to_filepaths[file_hash].append(str(filepath))

    files_to_delete = []
    for filepaths in hashes_to_filepaths.values():
        if len(filepaths) > 1:
            # Сортируем по длине пути, чтобы сохранить файл с самым коротким именем/путем
            filepaths.sort(key=len)
            files_to_delete.extend(filepaths[1:])

    if not files_to_delete:
        print("No duplicate photos found.")
        return

    print(f"\nFound {len(files_to_delete)} duplicate photos.")

    if delete:
        print("Starting deletion...")
        deleted_count = 0
        for filepath in tqdm(files_to_delete, desc="Deleting duplicates"):
            try:
                os.remove(filepath)
                deleted_count += 1
            except OSError as e:
                print(f"\nError deleting {filepath}: {e}")
        print(f"Deletion complete. Successfully deleted {deleted_count} files.")
    else:
        print("Running in DRY RUN mode. To delete files, run this script with the --delete flag.")
        for filepath in files_to_delete:
            print(f"  [DRY RUN] Would delete: {filepath}")

if __name__ == "__main__":
    # Скрипт ожидает, что он находится в папке scripts/
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    target_directory = PROJECT_ROOT / "data" / "3_for_training"

    # Проверяем, был ли передан флаг --delete из командной строки
    delete_mode_enabled = "--delete" in sys.argv
    
    find_and_delete_duplicate_photos(target_directory, delete=delete_mode_enabled)