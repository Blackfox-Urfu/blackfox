from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
import logging
import numpy as np
import cv2
from skimage.feature import hog
from typing import Optional
import time

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка моделей
try:
    # Текстовая модель
    model = joblib.load("randfor_model.pkl")
    vectorizer = joblib.load("randfor_vectorizer.pkl")
    logger.info("Текстовая модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки текстовой модели: {e}")
    model, vectorizer = None, None

try:
    # HOG модель
    hog_model = joblib.load("best_hog_model.pkl")
    hog_params = joblib.load("hog_params.pkl")
    logger.info("HOG модель и параметры успешно загружены")
except Exception as e:
    logger.error(f"Ошибка загрузки HOG модели: {e}")
    hog_model, hog_params = None, None

app = FastAPI()

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Time: {process_time:.2f}s"
    )
    return response

# Middleware для проверки размера файлов
@app.middleware("http")
async def check_file_size(request: Request, call_next):
    if request.method == "POST" and "/api/classify_image/" in request.url.path:
        content_length = int(request.headers.get("content-length", 0))
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        
        if content_length > MAX_FILE_SIZE:
            logger.warning(f"Попытка загрузки слишком большого файла: {content_length} bytes")
            raise HTTPException(
                status_code=413,
                detail=f"Файл слишком большой. Максимальный размер: {MAX_FILE_SIZE/(1024*1024):.0f}MB"
            )
    return await call_next(request)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Классы для запросов
class TextRequest(BaseModel):
    text: str

class ImageResponse(BaseModel):
    prediction: float
    is_nsfw: bool
    error: Optional[str] = None

# Текстовая классификация
@app.post("/api/classify_text/")
async def classify_text(request: TextRequest):
    if not model or not vectorizer:
        logger.error("Попытка использования текстовой модели, когда она не загружена")
        return {"error": "Текстовая модель не загружена"}
    
    try:
        logger.debug(f"Классификация текста: {request.text[:100]}...")
        prediction = float(model.predict_proba(vectorizer.transform([request.text]))[0][1])
        logger.info(f"Результат классификации текста: {prediction:.2f}")
        return {
            "prediction": prediction,
            "is_ad": prediction > 0.6
        }
    except Exception as e:
        logger.error(f"Ошибка классификации текста: {e}", exc_info=True)
        return {"error": str(e)}

# Классификация изображений
@app.post("/api/classify_image/", response_model=ImageResponse)
async def classify_image(file: UploadFile = File(...)):
    if not hog_model or not hog_params:
        logger.error("Попытка использования HOG модели, когда она не загружена")
        return ImageResponse(
            prediction=0.0,
            is_nsfw=False,
            error="HOG модель не загружена"
        )
    
    try:
        logger.info(f"Обработка изображения: {file.filename}")
        contents = await file.read()
        
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("Не удалось декодировать изображение")
            raise ValueError("Неверный формат изображения")
            
        features = hog(
            cv2.resize(img, (hog_params['resize'], hog_params['resize'])),
            orientations=hog_params['orientations'],
            pixels_per_cell=(hog_params['pixels_per_cell'], hog_params['pixels_per_cell']),
            cells_per_block=(hog_params['cells_per_block'], hog_params['cells_per_block']),
            block_norm='L2-Hys'
        )
        
        proba = 1 / (1 + np.exp(-hog_model.decision_function([features])[0]))
        logger.info(f"Результат классификации изображения: {proba:.2f}")
        
        return ImageResponse(
            prediction=float(proba),
            is_nsfw=proba > 0.5
        )
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {e}", exc_info=True)
        return ImageResponse(
            prediction=0.0,
            is_nsfw=False,
            error=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,  # Используем настройки логгирования из кода
        access_log=False  # Отключаем дублирование логов
    )