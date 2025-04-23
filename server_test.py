from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
import logging
import numpy as np
from typing import Optional
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import warnings

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка моделей
try:
    # Текстовая модель (игнорируем предупреждения о версиях)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = joblib.load("randfor_model.pkl")
        vectorizer = joblib.load("randfor_vectorizer.pkl")
    logger.info("Текстовая модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки текстовой модели: {e}")
    model, vectorizer = None, None

# Загрузка ResNet34 модели
try:
    # Инициализация модели
    resnet_model = models.resnet34(pretrained=False)
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    # Загрузка весов
    resnet_model.load_state_dict(torch.load("best_resnet34.pth", map_location=torch.device('cpu')))
    resnet_model.eval()
    logger.info("ResNet34 модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки ResNet34 модели: {e}")
    resnet_model = None

# Трансформации для изображений
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    if not resnet_model:
        logger.error("Попытка использования ResNet34 модели, когда она не загружена")
        return ImageResponse(
            prediction=0.0,
            is_nsfw=False,
            error="ResNet34 модель не загружена"
        )
    
    try:
        logger.info(f"Обработка изображения: {file.filename}")
        contents = await file.read()
        
        # Преобразование в PIL Image
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Применение трансформаций
        img_tensor = img_transform(img).unsqueeze(0)
        
        # Предсказание
        with torch.no_grad():
            prediction = resnet_model(img_tensor).item()
        
        logger.info(f"Результат классификации изображения: {prediction:.2f}")
        
        return ImageResponse(
            prediction=float(prediction),
            is_nsfw=prediction > 0.5
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
        log_config=None,
        access_log=False
    )