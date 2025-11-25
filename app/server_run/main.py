import fastapi
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import joblib
import uvicorn
import logging
from logging.handlers import TimedRotatingFileHandler
import uuid
import numpy as np
from typing import Optional, Union
import time
import os
import sys
import io
import re
import threading
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import pickle
import hashlib
import asyncio

# --- НОВЫЙ ИМПОРТ для метрик ---
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# --- Настройка путей и импортов ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# --- Пути к моделям ---
REKLAMA_MODELS_DIR = os.path.join(PROJECT_ROOT, "app", "learn", "reklama_classification_models")
RESNET_ARCH_DIR = os.path.join(PROJECT_ROOT, "app", "learn", "resnet_image")
MODEL_MULTIMODAL_DIR = os.path.join(PROJECT_ROOT, "model", "multimodal")
MODEL_TEXT_ONLY_DIR = os.path.join(PROJECT_ROOT, "model", "torch_text")
MODEL_RESNET_DIR = os.path.join(PROJECT_ROOT, "model", "resnet")

# --- Импорт архитектур моделей ---
if REKLAMA_MODELS_DIR not in sys.path:
    sys.path.insert(0, REKLAMA_MODELS_DIR)
if RESNET_ARCH_DIR not in sys.path:
    sys.path.insert(0, RESNET_ARCH_DIR)

try:
    from torch_models import MetaLearner
    from model_architecture import create_configurable_model
except ImportError as e:
    print(f"CRITICAL: Could not import MetaLearner or create_configurable_model. Error: {e}")
    sys.exit(1)

# --- Настройка логирования ---
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")
thread_local = threading.local()

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(thread_local, 'request_id', 'startup')
        return True

# 1. Создаем форматер и фильтр
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s')
request_id_filter = RequestIdFilter()

# 2. Создаем и настраиваем обработчик для консоли
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
console_handler.addFilter(request_id_filter)

# 3. Создаем и настраиваем обработчик для файла
file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight", interval=1, backupCount=7, encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING)
file_handler.addFilter(request_id_filter)

# 4. Настраиваем корневой логгер
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Важно: сначала очищаем старые обработчики, потом добавляем новые
root_logger.handlers.clear()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# 5. Получаем логгер для текущего модуля
logger = logging.getLogger(__name__)

# --- Инициализация FastAPI и Метрик ---
app = FastAPI(title="Black-Fox ML API")

# --- Настройки кэша и ограничений ---
NSFW_CACHE = {} 
# Ограничиваем одновременную работу с NSFW до 1 потока.
# Это спасет память и CPU.
nsfw_semaphore = asyncio.Semaphore(1)

# --- НОВЫЙ БЛОК: Настройка метрик Prometheus ---
PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total number of predictions by model and result.",
    ["model_type", "result"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken for a model prediction.",
    ["model_type"]
)
# Инструментируем приложение для сбора стандартных метрик и открытия эндпоинта /metrics
Instrumentator().instrument(app).expose(app)
# =================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4().hex[:8])
    thread_local.request_id = request_id
    logger.info(f"Request started: {request.method} {request.url.path} from {request.client.host}")
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request finished: {response.status_code} in {process_time:.4f}s")
    setattr(thread_local, 'request_id', None)
    return response

# --- Определение архитектуры моделей ---
class AdvancedTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers=[512, 256, 128], num_classes=2, dropout=0.3, activation='relu', use_batch_norm=True):
        super(AdvancedTextClassifier, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm: layers.append(nn.BatchNorm1d(hidden_size))
            if activation == 'relu': layers.append(nn.ReLU())
            elif activation == 'leaky_relu': layers.append(nn.LeakyReLU(0.1))
            else: layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, num_classes)
    def forward(self, x):
        current_batch_size = x.size(0)
        processed_x = x
        for layer in self.hidden_layers:
            if isinstance(layer, nn.BatchNorm1d) and current_batch_size <= 1: continue
            processed_x = layer(processed_x)
        return self.output_layer(processed_x)

# --- Классы-обертки для классификаторов ---
class TextOnlyClassifier:
    def __init__(self):
        self.model: Optional[AdvancedTextClassifier] = None
        self.vectorizer = None
        self.threshold = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # === НОВЫЙ ПАРАМЕТР: Температура для смягчения предсказаний ===
        self.temperature = 2.0 # Значение > 1. Можно подобрать (например, 1.5, 2.0, 2.5)
        logger.info(f"TextOnlyClassifier initialized. Using device: {self.device}, Temperature: {self.temperature}")

    def load(self, model_dir: str) -> bool:
        # ... (код загрузки остается без изменений)
        model_path = os.path.join(model_dir, 'best_final_model.pth')
        vectorizer_path = os.path.join(model_dir, 'final_vectorizer.pkl')
        try:
            if not all(os.path.exists(p) for p in [model_path, vectorizer_path]):
                logger.error(f"One or more text-only model files not found in {model_dir}")
                return False
            self.vectorizer = joblib.load(vectorizer_path)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model_config = checkpoint['model_config']
            self.threshold = checkpoint.get('threshold', 0.5)
            self.model = AdvancedTextClassifier(
                input_size=model_config['input_size'], hidden_layers=model_config['hidden_layers'],
                dropout=model_config['dropout'], activation=model_config['activation'],
                use_batch_norm=model_config['use_batch_norm']
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            logger.info(f"Text-only model loaded from {model_path} with threshold {self.threshold:.4f}")
            return True
        except Exception as e:
            logger.error(f"Error loading text-only model: {str(e)}", exc_info=True)
            self.model, self.vectorizer = None, None
            return False

    def predict(self, text: str) -> tuple[float, bool]:
        if not self.model or not self.vectorizer: raise RuntimeError("Text-only model is not loaded.")
        vector = self.vectorizer.transform([text]).toarray().astype(np.float32)
        tensor_input = torch.tensor(vector, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor_input)
            
            # === ИСПРАВЛЕНИЕ: Применяем температурное масштабирование ===
            # Делим логиты на температуру, чтобы сделать softmax менее "резким"
            scaled_outputs = outputs / self.temperature
            probabilities = torch.softmax(scaled_outputs, dim=1)
            # ==========================================================

            prob_ad = probabilities[0][1].item()
        is_ad = prob_ad > self.threshold
        return prob_ad, is_ad

class MultimodalClassifier:
    def __init__(self):
        self.model: Optional[MetaLearner] = None
        self.text_vectorizer = None
        self.feature_scaler = None
        self.image_transform: Optional[transforms.Compose] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"MultimodalClassifier initialized. Using device: {self.device}")

    def load(self, model_dir: str) -> bool:
        model_path = os.path.join(model_dir, 'best_model.pth'); vectorizer_path = os.path.join(model_dir, 'text_vectorizer.pkl'); scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        try:
            if not all(os.path.exists(p) for p in [model_path, vectorizer_path, scaler_path]): logger.error("One or more multimodal model files not found!"); return False
            self.text_vectorizer = joblib.load(vectorizer_path); self.feature_scaler = joblib.load(scaler_path)
            logger.info("Multimodal vectorizer and scaler loaded.")
            # Важно: Убеждаемся, что количество признаков совпадает
            text_input_size = self.text_vectorizer.max_features
            features_input_size = self.feature_scaler.n_features_in_
            
            self.model = MetaLearner(text_input_size=text_input_size, features_input_size=features_input_size).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False)); self.model.eval()
            logger.info(f"Multimodal model (MetaLearner) loaded from {model_path}.")
            self.image_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            return True
        except Exception as e:
            logger.error(f"Error loading multimodal model: {str(e)}", exc_info=True)
            self.model, self.text_vectorizer, self.feature_scaler = None, None, None
            return False

    # === ИСПРАВЛЕННЫЙ МЕТОД ===
    def _extract_features(self, text: str, has_image: bool) -> np.ndarray:
        # Эта логика должна максимально точно повторять логику из `torch_multimodal.py`
        text_length = len(text)
        # Считаем ссылки более надежным способом
        link_count = len(re.findall(r'http[s]?://\S+', text))
        mention_count = text.count('@')
        hashtag_count = text.count('#')
        # В API у нас только одно изображение, так что это будет 1 или 0
        attachment_count = 1 if has_image else 0
        
        # Создаем вектор признаков в том же порядке, что и при обучении
        features = [text_length, link_count, mention_count, hashtag_count, attachment_count]
        
        return np.array(features).reshape(1, -1)

    def predict(self, text: str, image: Optional[Image.Image]) -> tuple[float, bool]:
        if not all([self.model, self.text_vectorizer, self.feature_scaler, self.image_transform]): raise RuntimeError("Multimodal model is not loaded.")
        
        text_vector = self.text_vectorizer.transform([text]).toarray().astype(np.float32)
        
        # Используем исправленный метод извлечения признаков
        features_vector_raw = self._extract_features(text, has_image=(image is not None))
        features_vector = self.feature_scaler.transform(features_vector_raw).astype(np.float32)

        batch = {
            'text': torch.tensor(text_vector, dtype=torch.float32).to(self.device), 
            'features': torch.tensor(features_vector, dtype=torch.float32).to(self.device), 
            'labels': torch.tensor([0]).to(self.device) # Лейбл-плейсхолдер
        }

        if image:
            try:
                image_tensor = self.image_transform(image.convert('RGB')).unsqueeze(0)
                batch['images'] = image_tensor.to(self.device)
                batch['image_indices'] = torch.tensor([0], dtype=torch.long).to(self.device)
            except Exception as e: logger.warning(f"Could not process image: {e}")
        
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            prob_ad = probabilities[0][1].item()
            
        return prob_ad, prob_ad > 0.5

class ImageClassifier:
    def __init__(self):
        self.model = None
        self.transform: Optional[transforms.Compose] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = 0.5
        logger.info(f"ImageClassifier (NSFW) initialized. Using device: {self.device}")

    def load(self, model_path: str) -> bool:
        try:
            params_path = os.path.join(MODEL_RESNET_DIR, 'best_optuna_params.pkl')
            if not all(os.path.exists(p) for p in [model_path, params_path]):
                logger.error(f"NSFW model or params file not found in {MODEL_RESNET_DIR}")
                return False
            with open(params_path, 'rb') as f:
                best_params = pickle.load(f)
            self.model = create_configurable_model(best_params)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.model.to(self.device)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info(f"Image model (NSFW, base: {best_params.get('base_model', 'unknown')}) loaded from {model_path}.")
            return True
        except Exception as e:
            logger.error(f"Error loading image model (NSFW): {str(e)}", exc_info=True)
            self.model, self.transform = None, None
            return False

# --- Инициализация всех классификаторов ---
multimodal_classifier = MultimodalClassifier()
text_only_classifier = TextOnlyClassifier()
nsfw_image_classifier = ImageClassifier()

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    multimodal_loaded = multimodal_classifier.load(model_dir=MODEL_MULTIMODAL_DIR)
    if not multimodal_loaded: logger.warning("MULTIMODAL ad classification model FAILED to load.")
    text_only_loaded = text_only_classifier.load(model_dir=MODEL_TEXT_ONLY_DIR)
    if not text_only_loaded: logger.warning("TEXT-ONLY ad classification model FAILED to load.")
    nsfw_model_loaded = nsfw_image_classifier.load(model_path=os.path.join(MODEL_RESNET_DIR, 'best_resnet_state_dict.pth'))
    if not nsfw_model_loaded: logger.warning("NSFW classification model FAILED to load.")
    logger.info("Application startup complete.")

# --- Модели ответа Pydantic ---
class AdPredictionResponse(BaseModel):
    prediction_prob_ad: float
    is_ad: bool
    confidence: float
    error: Optional[str] = None

class NsfwPredictionResponse(BaseModel):
    prediction_prob_nsfw: float
    is_nsfw: bool
    confidence: float
    error: Optional[str] = None

class TextRequest(BaseModel):
    text: str

# --- Эндпоинты API ---
@app.post("/api/classify_message/", response_model=AdPredictionResponse)
async def classify_message_endpoint(text: str = Form(""), image: Optional[UploadFile] = File(None)):
    if not multimodal_classifier.model: raise HTTPException(status_code=503, detail="Multimodal ad model not available.")
    try:
        pil_image, image_info = None, "no image"
        if image and image.filename:
            image_info = f"image present (filename: {image.filename})"
            contents = await image.read()
            if contents: pil_image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Classifying multimodal. Text len: {len(text)}, {image_info}.")
        if not text.strip() and not pil_image: return AdPredictionResponse(prediction_prob_ad=0.0, is_ad=False, confidence=1.0, error="No content")

        # МОНИТОРИНГ
        with PREDICTION_LATENCY.labels(model_type="multimodal").time():
            prob_ad, is_ad = multimodal_classifier.predict(text=text, image=pil_image)
        PREDICTION_COUNTER.labels(model_type="multimodal", result="ad" if is_ad else "not_ad").inc()

        logger.info(f"Multimodal result: is_ad={is_ad}, prob={prob_ad:.4f}")
        return AdPredictionResponse(prediction_prob_ad=prob_ad, is_ad=is_ad, confidence=prob_ad if is_ad else (1 - prob_ad))
    except Exception as e: logger.error(f"Error in multimodal endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify_text/", response_model=AdPredictionResponse)
async def classify_text_endpoint(request: TextRequest):
    if not text_only_classifier.model: raise HTTPException(status_code=503, detail="Text-only ad model not available.")
    try:
        if not request.text or not request.text.strip(): return AdPredictionResponse(prediction_prob_ad=0.0, is_ad=False, confidence=1.0, error="Empty text")
        
        logger.info(f"Classifying text-only. Text len: {len(request.text)}.")
        
        # МОНИТОРИНГ
        with PREDICTION_LATENCY.labels(model_type="text_only").time():
            prob_ad, is_ad = text_only_classifier.predict(request.text)
        PREDICTION_COUNTER.labels(model_type="text_only", result="ad" if is_ad else "not_ad").inc()

        logger.info(f"Text-only result: is_ad={is_ad}, prob={prob_ad:.4f}")
        return AdPredictionResponse(prediction_prob_ad=prob_ad, is_ad=is_ad, confidence=prob_ad if is_ad else (1 - prob_ad))
    except Exception as e: logger.error(f"Error in text-only endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify_nsfw_image/", response_model=NsfwPredictionResponse)
async def classify_nsfw_image_endpoint(file: UploadFile = File(...)):
    if not nsfw_image_classifier.model:
        raise HTTPException(status_code=503, detail="NSFW model not available.")

    # ВАЖНО: СЕМАФОР ДОЛЖЕН БЫТЬ ТУТ, ДО ЧТЕНИЯ ФАЙЛА
    async with nsfw_semaphore:
        try:
            contents = await file.read() # Читаем только когда подошла очередь
            if not contents:
                raise HTTPException(status_code=400, detail="Empty file")

            # Считаем хэш
            file_hash = hashlib.md5(contents).hexdigest()

            # Проверяем кэш
            if file_hash in NSFW_CACHE:
                logger.info(f"Cache HIT for {file.filename} ({file_hash})")
                return NSFW_CACHE[file_hash]

            # 4. Если в кэше нет - запускаем нейросеть
            logger.info(f"Processing NSFW. Filename: {file.filename} Hash: {file_hash}")
            
            loop = asyncio.get_event_loop()
            
            def process_image():
                # Конвертация и инференс - тяжелые операции
                try:
                    img = Image.open(io.BytesIO(contents)).convert('RGB')
                    img_tensor = nsfw_image_classifier.transform(img).unsqueeze(0).to(nsfw_image_classifier.device)
                    
                    with torch.no_grad():
                        logits = nsfw_image_classifier.model(img_tensor)
                        prob_nsfw = torch.sigmoid(logits).item()
                    return prob_nsfw
                except Exception as e:
                    logger.error(f"Error inside processing thread: {e}")
                    raise e

            # Замер времени
            with PREDICTION_LATENCY.labels(model_type="nsfw").time():
                prob_nsfw = await loop.run_in_executor(None, process_image)

            is_nsfw = prob_nsfw > nsfw_image_classifier.threshold
            PREDICTION_COUNTER.labels(model_type="nsfw", result="nsfw" if is_nsfw else "sfw").inc()

            response_data = NsfwPredictionResponse(
                prediction_prob_nsfw=prob_nsfw,
                is_nsfw=is_nsfw,
                confidence=prob_nsfw if is_nsfw else (1 - prob_nsfw)
            )

            logger.info(f"NSFW result: is_nsfw={is_nsfw}, prob={prob_nsfw:.4f}")

            # 5. Сохраняем результат
            if len(NSFW_CACHE) > 2000: # Лимит кэша
                NSFW_CACHE.clear()
            NSFW_CACHE[file_hash] = response_data
            
            return response_data

        except UnidentifiedImageError:
            logger.warning(f"Could not identify image format for file: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid or unsupported image format.")
        except Exception as e:
            logger.error(f"Error in NSFW endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    logger.info("Health check requested.")
    return {
        "status": "healthy", 
        "multimodal_model_loaded": multimodal_classifier.model is not None, 
        "text_only_model_loaded": text_only_classifier.model is not None, 
        "nsfw_model_loaded": nsfw_image_classifier.model is not None, 
        "device": str(multimodal_classifier.device)
    }

# Этот блок используется только для локальной разработки.
# В продакшене запуск осуществляется через systemd, который вызывает uvicorn напрямую.
if __name__ == "__main__":
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        workers=1, 
        log_config=None,
        ssl_keyfile="./key.pem",      # Укажите путь к вашему локальному ключу
        ssl_certfile="./cert.pem" # Укажите путь к вашему локальному сертификату
    )