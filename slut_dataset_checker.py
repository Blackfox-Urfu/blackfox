import ollama # requires-python = "=3.12.3"
import os
import shutil
from tqdm import tqdm  

# Пути к данным
SLUT_DIR = "data/slut"  # Исходная папка с NSFW данными
TO_DELETE_DIR = "data/to_delete_from_nsfw"  # Куда копировать не-NSFW фото

# Создаем папку для удаляемых файлов
os.makedirs(TO_DELETE_DIR, exist_ok=True)

slut_promt = """Analyze this image as a strict content moderator. Determine if it matches ALL the key features of Russian/Ukrainian avatars for dating bots.:

CRITERIA (answer "yes" only if you meet 2+ points):
1. POSE: butt/chest in the spotlight (bottom selfie, arching back)
2. CLOTHING: Transparent/partially removed underwear, lowered panties/bra
3. FOCUS: The crotch/neckline is in the center of the frame, with the face visible by less than 30%.
4. BANKING DETAILS: A bed/sofa in a sexy frame
5. CLOTHING: Only bras/ thongs with visible contours of the areola/pubis

CRITERIA ARE NOT TAKEN INTO ACCOUNT (automatic "no", if any):
- Fashionable full-length shots
- Bikinis for the beach/pool without an emphasis on the genitals
- Artistic nudity in neutral poses
- Gym/sports wear without an emphasis on the crotch

Answer ONLY "yes" (corresponds to the avatar of the whore-bot or this is NSFW) or "no" (a regular image, only if you are 1000 percent sure, as if she were a nun)."""

def is_nsfw(image_path: str) -> bool:
    """Определяет, является ли изображение NSFW.
    Возвращает True если NSFW (оставляем), False если нужно удалить."""
    prompt = slut_promt

    try:
        response = ollama.generate(
            model="gemma3:4b",
            prompt=prompt,
            images=[image_path],
            options={"temperature": 0.1}  
        )
        return "yes" in response["response"].lower().strip()
    except Exception as e:
        print(f"\n[ERROR] Ошибка при анализе {os.path.basename(image_path)}: {e}")
        return True  # При ошибке считаем файл NSFW (не удаляем)

print(f"Начинаем проверку папки '{SLUT_DIR}'...")

# Фильтрация с прогресс-баром
for img_file in tqdm([f for f in os.listdir(SLUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]):
    img_path = os.path.join(SLUT_DIR, img_file)
    
    if not is_nsfw(img_path):
        try:
            # Копируем в папку для удаления (без перезаписи)
            dest_path = os.path.join(TO_DELETE_DIR, img_file)
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
        except Exception as e:
            print(f"\n[COPY ERROR] Не удалось скопировать {img_file}: {e}")

print("\nГотово! Результаты:")
print(f"- Всего проверено: {len(os.listdir(SLUT_DIR))} файлов")
print(f"- Предлагается к удалению: {len(os.listdir(TO_DELETE_DIR))} файлов")
print(f"\nФайлы для проверки находятся в '{TO_DELETE_DIR}'")