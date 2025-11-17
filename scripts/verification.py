import os
import cv2
from tqdm import tqdm
import multiprocessing

def check_image_worker(filepath):
    """
    Рабочая функция для одного процесса.
    Проверяет один файл и возвращает его путь, если он "битый".
    В противном случае возвращает None.
    """
    try:
        # Проверяем, что файл не пустой, чтобы избежать лишних вызовов imread
        if os.path.getsize(filepath) == 0:
            return filepath

        img = cv2.imread(filepath)
        if img is None:
            return filepath  # Возвращаем путь, если файл не читается
    except Exception:
        return filepath # Возвращаем путь при любой ошибке чтения
    return None

def verify_images_parallel(directory):
    """
    Параллельно проверяет все изображения в директории и удаляет те,
    которые не могут быть открыты OpenCV.
    """
    print(f"Collecting files from: {directory}...")
    filepaths_to_check = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths_to_check.append(os.path.join(root, filename))

    if not filepaths_to_check:
        print(f"No image files found in {directory}.")
        return

    print(f"Found {len(filepaths_to_check)} images to verify. Starting parallel processing...")

    bad_files = []
    # Определяем количество процессов (можно взять все доступные ядра)
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes.")

    # Создаем пул процессов
    with multiprocessing.Pool(processes=num_processes) as pool:
        # imap_unordered более эффективен для задач, где порядок выполнения не важен
        # tqdm будет отображать прогресс по мере завершения задач
        results_iterator = pool.imap_unordered(check_image_worker, filepaths_to_check)
        
        for result in tqdm(results_iterator, total=len(filepaths_to_check), desc=f"Verifying {os.path.basename(directory)}"):
            if result is not None:
                bad_files.append(result)

    if not bad_files:
        print(f"\nNo bad files found in {directory}.")
        return

    # Удаление "битых" файлов
    print(f"\nFound {len(bad_files)} bad files. Deleting them...")
    deleted_count = 0
    for filepath in bad_files:
        try:
            os.remove(filepath)
            deleted_count += 1
        except OSError as e:
            print(f"  Error deleting {filepath}: {e}")
    print(f"Successfully deleted {deleted_count} files.")


if __name__ == '__main__':
    # --- Укажите пути к вашим папкам с данными ---
    SLUT_DATA_DIR = 'data/reddit/nsfw_images'
    REGULAR_DATA_DIR = 'data/reddit/sfw_images'

    print("--- Verifying NSFW images (in parallel) ---")
    verify_images_parallel(SLUT_DATA_DIR)

    print("\n--- Verifying SFW images (in parallel) ---")
    verify_images_parallel(REGULAR_DATA_DIR)

    print("\nVerification complete.")