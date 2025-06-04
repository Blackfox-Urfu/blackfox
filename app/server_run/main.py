from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import joblib
import uvicorn
import logging
import numpy as np
from typing import Optional, List
import time
import os
import sys
import io

# --- PyTorch, torchvision, PIL для классификации изображений ---
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
# --- Конец импортов для изображений ---

# --- НАЧАЛО ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТURY (остается из предыдущей версии) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

APP_DIR = os.path.join(PROJECT_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

MODEL_FILES_TEXT_DIR = os.path.join(PROJECT_ROOT, "model", "torch_text")
# --- Путь к моделям ResNet ---
MODEL_FILES_RESNET_DIR = os.path.join(PROJECT_ROOT, "model", "resnet")

from learn.torch_text.model_architecture import AdvancedTextClassifier
# --- КОНЕЦ ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Класс TextClassifier (без изменений) ---
class TextClassifier:
    def __init__(self):
        self.model: Optional[AdvancedTextClassifier] = None
        self.vectorizer = None
        self.threshold = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"TextClassifier initialized. Using device: {self.device}")

    def load(self, model_checkpoint_path: str, vectorizer_path: str) -> bool:
        try:
            if not os.path.exists(model_checkpoint_path):
                logger.error(f"Text model checkpoint file not found: {model_checkpoint_path}")
                return False
            if not os.path.exists(vectorizer_path):
                logger.error(f"Vectorizer file not found: {vectorizer_path}")
                return False

            self.vectorizer = joblib.load(vectorizer_path)
            logger.info(f"Vectorizer loaded from {vectorizer_path}")

            checkpoint = torch.load(model_checkpoint_path, map_location=self.device, weights_only=False)
            model_config = checkpoint['model_config']
            self.threshold = checkpoint.get('threshold', self.threshold)
            logger.info(f"Text model using threshold: {self.threshold}")

            self.model = AdvancedTextClassifier(
                input_size=model_config['input_size'],
                hidden_layers=model_config['hidden_layers'],
                num_classes=model_config.get('num_classes', 2),
                dropout=model_config['dropout'],
                activation=model_config['activation'],
                use_batch_norm=model_config['use_batch_norm']
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            logger.info(f"Text model loaded from {model_checkpoint_path} and configured successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading text model: {str(e)}", exc_info=True)
            self.model, self.vectorizer = None, None
            return False

# --- Класс ImageClassifier (НОВЫЙ) ---
class ImageClassifier:
    def __init__(self):
        self.model: Optional[models.ResNet] = None
        self.transform: Optional[transforms.Compose] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = 0.5 # Порог для NSFW классификации по умолчанию
        logger.info(f"ImageClassifier initialized. Using device: {self.device}")

    def load(self, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                logger.error(f"Image model file not found: {model_path}")
                return False

            # Инициализация модели ResNet34 с кастомной головой
            self.model = models.resnet34(pretrained=False) # pretrained=False, т.к. загружаем все веса
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

            # Загрузка весов
            # map_location=self.device гарантирует, что тензоры загрузятся на правильное устройство
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device) # Убедимся, что вся модель на нужном устройстве
            self.model.eval() # Переключение модели в режим оценки

            # Определение трансформаций для изображений
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info(f"Image model (ResNet34) loaded from {model_path} and configured successfully.")
            # Можно добавить загрузку порога из файла конфигурации, если это необходимо
            # logger.info(f"Image model using threshold: {self.threshold}")
            return True
        except Exception as e:
            logger.error(f"Error loading image model: {str(e)}", exc_info=True)
            self.model = None
            self.transform = None
            return False

# --- Инициализация классификаторов ---
text_classifier = TextClassifier()
image_classifier = ImageClassifier() # Новый классификатор

# Загрузка моделей при старте
text_model_loaded = text_classifier.load(
    model_checkpoint_path=os.path.join(MODEL_FILES_TEXT_DIR, 'best_final_model.pth'),
    vectorizer_path=os.path.join(MODEL_FILES_TEXT_DIR, 'final_vectorizer.pkl')
)
if not text_model_loaded:
    logger.warning("Text classification model FAILED to load.")

image_model_loaded = image_classifier.load(
    model_path=os.path.join(MODEL_FILES_RESNET_DIR, 'best_resnet.pth')
)
if not image_model_loaded:
    logger.warning("Image classification model (ResNet34) FAILED to load.")

app = FastAPI()

# --- Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s"
    )
    return response

# Middleware для проверки размера файлов (НОВЫЙ)
@app.middleware("http")
async def check_file_size(request: Request, call_next):
    # Проверяем только для эндпоинта загрузки изображений
    if request.method == "POST" and request.url.path == "/api/classify_image/":
        content_length_header = request.headers.get("content-length")
        if content_length_header:
            try:
                content_length = int(content_length_header)
                MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB (увеличил с 10МБ)
                
                if content_length > MAX_FILE_SIZE:
                    logger.warning(f"Attempt to upload too large file: {content_length} bytes. Max: {MAX_FILE_SIZE} bytes.")
                    # Немедленно возвращаем ошибку, не вызывая call_next
                    # FastAPI автоматически сконвертирует HTTPException в JSON ответ
                    raise HTTPException(
                        status_code=413, # Payload Too Large
                        detail=f"Файл слишком большой. Максимальный размер: {MAX_FILE_SIZE/(1024*1024):.0f}MB."
                    )
            except ValueError:
                logger.warning(f"Invalid content-length header: {content_length_header}")
                raise HTTPException(status_code=400, detail="Invalid content-length header.")
    
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модели запросов/ответов ---
class TextRequest(BaseModel):
    text: str

class TextPredictionResponse(BaseModel): # Переименовал для ясности
    prediction_prob_ad: float
    is_ad: bool
    confidence: float
    error: Optional[str] = None

class ImagePredictionResponse(BaseModel): # Новый для изображений
    prediction_prob_nsfw: float # Вероятность того, что это NSFW
    is_nsfw: bool
    confidence: float
    error: Optional[str] = None

# --- API эндпоинты ---
@app.post("/api/classify_text/", response_model=TextPredictionResponse)
async def classify_text_endpoint(request: TextRequest):
    if not text_classifier.model or not text_classifier.vectorizer:
        logger.error("Text model not loaded for /api/classify_text/")
        raise HTTPException(status_code=503, detail="Text classification model is not available.")
    
    try:
        if not request.text.strip():
            return TextPredictionResponse(
                prediction_prob_ad=0.0, is_ad=False, confidence=1.0, error="Input text is empty"
            )
        vector = text_classifier.vectorizer.transform([request.text]).toarray()
        if vector.shape[0] == 0:
             return TextPredictionResponse(prediction_prob_ad=0.0, is_ad=False, confidence=1.0, error="Text could not be vectorized")
        
        tensor_input = torch.tensor(vector[0].astype(np.float32), dtype=torch.float32).unsqueeze(0).to(text_classifier.device)
        
        with torch.no_grad():
            outputs = text_classifier.model(tensor_input)
            probabilities = torch.softmax(outputs, dim=1)
            prob_ad = probabilities[0][1].item()
        
        is_ad_prediction = prob_ad > text_classifier.threshold
        return TextPredictionResponse(
            prediction_prob_ad=prob_ad,
            is_ad=is_ad_prediction,
            confidence=prob_ad if is_ad_prediction else (1 - prob_ad)
        )
    except Exception as e:
        logger.error(f"Error during text classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during text classification: {str(e)}")

# Эндпоинт для классификации изображений (НОВЫЙ)
@app.post("/api/classify_image/", response_model=ImagePredictionResponse)
async def classify_image_endpoint(file: UploadFile = File(...)):
    if not image_classifier.model or not image_classifier.transform:
        logger.error("Image model not loaded for /api/classify_image/")
        raise HTTPException(status_code=503, detail="Image classification model is not available.")

    try:
        logger.debug(f"Processing image: {file.filename}, content_type: {file.content_type}")
        contents = await file.read()
        
        # Проверка, что файл не пустой
        if not contents:
            logger.warning(f"Uploaded file {file.filename} is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Преобразование в PIL Image
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as pil_e:
            logger.error(f"Error opening image {file.filename} with PIL: {pil_e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Could not process image file. Ensure it's a valid image format. Error: {str(pil_e)}")

        # Применение трансформаций
        img_tensor = image_classifier.transform(img).unsqueeze(0).to(image_classifier.device)
        
        # Предсказание
        with torch.no_grad():
            # Модель ResNet с Sigmoid на выходе уже дает вероятность 0-1
            prob_nsfw = image_classifier.model(img_tensor).item() 
        
        is_nsfw_prediction = prob_nsfw > image_classifier.threshold
        
        logger.info(f"Image classification result for {file.filename}: prob_nsfw={prob_nsfw:.4f}, is_nsfw={is_nsfw_prediction}")
        
        return ImagePredictionResponse(
            prediction_prob_nsfw=prob_nsfw,
            is_nsfw=is_nsfw_prediction,
            confidence=prob_nsfw if is_nsfw_prediction else (1 - prob_nsfw)
        )
    except HTTPException: # Перехватываем HTTPException, чтобы не попасть в общий Exception ниже
        raise
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during image processing: {str(e)}")


# Health check (обновленный)
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "text_model_loaded": text_classifier.model is not None and text_classifier.vectorizer is not None,
        "text_model_device": str(text_classifier.device) if text_classifier.device else "N/A",
        "image_model_loaded": image_classifier.model is not None and image_classifier.transform is not None,
        "image_model_device": str(image_classifier.device) if image_classifier.device else "N/A",
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, # Отключите для продакшена
        log_level="info",
        access_log=False # Логи доступа уже обрабатываются middleware
    )