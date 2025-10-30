import subprocess
import os
import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
dotenv_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=dotenv_path)

SERVER_USER = os.getenv("SERVER_USER")
SERVER_HOST = os.getenv("SERVER_HOST")
SERVER_PROJECT_PATH = os.getenv("SERVER_PROJECT_PATH")

if not all([SERVER_USER, SERVER_HOST, SERVER_PROJECT_PATH]):
    print("❌ Ошибка: Не все переменные окружения определены в .env")
    print("   Пожалуйста, убедитесь, что в файле .env в корне проекта есть SERVER_USER, SERVER_HOST и SERVER_PROJECT_PATH.")
    sys.exit(1)

print(f"✅ Конфигурация загружена. Сервер: {SERVER_USER}@{SERVER_HOST}")

# --- УПРАВЛЕНИЕ ЭТАПАМИ ---
PREPARE_DATA = True
CLEAN_DUPLICATES = True
TRAIN_TEXT_MODEL = True
TRAIN_NSFW_MODEL = True
TRAIN_MULTIMODAL_MODEL = True
DEPLOY_TO_SERVER = True

# --- ПУТИ ---
UNIFIED_ADS_FILE = PROJECT_ROOT / "data" / "2_interim_unified" / "ads_unified.json"
UNIFIED_NON_ADS_FILE = PROJECT_ROOT / "data" / "2_interim_unified" / "non_ads_unified.json"
TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "3_for_training"

def run_command(command, cwd="."):
    """Выполняет команду и печатает ее вывод в реальном времени."""
    print(f"\n{'='*20}\n[RUNNING IN {cwd}]: {' '.join(command)}\n{'='*20}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace', # Добавлено для избежания ошибок кодировки
        cwd=cwd
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"\n{'!'*20}\n[ERROR]: Command failed with exit code {process.returncode}\n{'!'*20}")
        sys.exit(process.returncode)
    print(f"\n{'-'*20}\n[SUCCESS]: Command finished.\n{'-'*20}")

def sort_images_for_training():
    """Читает унифицированные JSON и копирует изображения в папки для обучения."""
    print("\nStep 1.2: Sorting images into training folders...")
    
    folders = {
        "reklama": TRAINING_DATA_DIR / "multimodal" / "reklama",
        "nereklama": TRAINING_DATA_DIR / "multimodal" / "nereklama",
        "slut": TRAINING_DATA_DIR / "nsfw_images" / "slut",
        "regular": TRAINING_DATA_DIR / "nsfw_images" / "regular"
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    def process_file(filepath, multi_dest, nsfw_dest):
        if not filepath.exists():
            print(f"  Warning: {filepath.name} not found, skipping.")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        messages = data.get("messages", [])
        print(f"  Processing {len(messages)} messages from {filepath.name}...")
        for msg in tqdm(messages, desc=f"Copying images from {filepath.name}"):
            for att in msg.get("attachments", []):
                if att.get("type") == "photo" and att.get("is_valid"):
                    source_path = PROJECT_ROOT / att["path"]
                    if source_path.exists():
                        try:
                            shutil.copy(source_path, multi_dest)
                            shutil.copy(source_path, nsfw_dest)
                        except shutil.SameFileError:
                            # Файл уже на месте, это нормально
                            pass
                        except Exception as e:
                            print(f"\nError copying {source_path}: {e}")

    process_file(UNIFIED_ADS_FILE, folders["reklama"], folders["slut"])
    process_file(UNIFIED_NON_ADS_FILE, folders["nereklama"], folders["regular"])
    print("Image sorting finished.")

# --- ДОБАВЛЕН НЕДОСТАЮЩИЙ КОД ---
def upload_artifacts_to_server():
    """Загружает артефакты моделей на сервер с помощью rsync."""
    print("\nUploading model artifacts to server...")
    source_path = PROJECT_ROOT / "model"
    # Убедимся, что папка model существует локально
    if not source_path.is_dir():
        print(f"Error: Local model directory not found at {source_path}. Nothing to upload.")
        return

    destination_path = f"{SERVER_USER}@{SERVER_HOST}:{SERVER_PROJECT_PATH}/"
    
    rsync_command = [
        "rsync",
        "-avz",
        "--delete",
        str(source_path) + "/",
        destination_path + "model/"
    ]
    
    run_command(rsync_command, cwd=str(PROJECT_ROOT))

# --- ДОБАВЛЕН НЕДОСТАЮЩИЙ КОД ---
def restart_server_service():
    """Перезапускает uvicorn.service на сервере по SSH."""
    print("\nRestarting uvicorn service on the server...")
    
    # Команда для SSH. `&& sleep 5` дает сервису время на запуск перед проверкой статуса.
    ssh_command_script = "sudo systemctl restart uvicorn.service && sleep 5 && sudo systemctl status uvicorn.service"
    
    ssh_command = [
        "ssh",
        f"{SERVER_USER}@{SERVER_HOST}",
        ssh_command_script
    ]
    run_command(ssh_command)

def main():
    """Главная функция для оркестрации подготовки, обучения и развертывания."""
    
    if PREPARE_DATA:
        print("\n--- Starting Data Preparation Stage ---")
        run_command([sys.executable, "-m", "scripts.merge"], cwd=str(PROJECT_ROOT))
        sort_images_for_training()
        if CLEAN_DUPLICATES:
            run_command([sys.executable, "-m", "scripts.dubl_check", "--delete"], cwd=str(PROJECT_ROOT))

    print("\n--- Starting Model Training Stage ---")
    if TRAIN_TEXT_MODEL:
        run_command([sys.executable, "-m", "app.learn.simple_text_torch.torch_text"], cwd=str(PROJECT_ROOT))
    if TRAIN_NSFW_MODEL:
        run_command([sys.executable, "-m", "app.learn.resnet_image.resnet_learn_slut_detector"], cwd=str(PROJECT_ROOT))
    if TRAIN_MULTIMODAL_MODEL:
        run_command([sys.executable, "-m", "app.learn.reklama_classification_models.torch_multimodal"], cwd=str(PROJECT_ROOT))

    if DEPLOY_TO_SERVER:
        print("\n\n--- All Training Processes Finished, Starting Deployment ---")
        upload_artifacts_to_server()
        restart_server_service()
        print("\n\n--- Deployment Finished Successfully! ---")
    else:
        print("\n\n--- All Training Processes Finished. Deployment skipped by configuration. ---")

if __name__ == "__main__":
    main()