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
from typing import Optional, Union, List
import time
import os
import sys
import io
import re
import threading
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

# --- Настройка путей и импортов ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# --- Пути к моделям ---
REKLAMA_MODELS_DIR = os.path.join(PROJECT_ROOT, "app", "learn", "reklama_classification_models")
MODEL_MULTIMODAL_DIR = os.path.join(PROJECT_ROOT, "model", "multimodal")
MODEL_TEXT_ONLY_DIR = os.path.join(PROJECT_ROOT, "model", "torch_text")
MODEL_RESNET_DIR = os.path.join(PROJECT_ROOT, "model", "resnet")

# --- Импорт архитектур моделей ---
if REKLAMA_MODELS_DIR not in sys.path:
    sys.path.insert(0, REKLAMA_MODELS_DIR)
try:
    from torch_models import MetaLearner
except ImportError:
    print(f"CRITICAL: Could not import MetaLearner from {REKLAMA_MODELS_DIR}")
    sys.exit(1)

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

# --- Настройка логирования ---
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")
thread_local = threading.local()
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(thread_local, 'request_id', 'startup'); return True
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s')
# Обработчик для консоли - оставляем INFO для отладки
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO) # <-- Уровень для консоли
# Обработчик для файла - ставим WARNING, чтобы не писать лишнего
file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight", interval=1, backupCount=7, encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING) # <-- Уровень для файла
# Настройка корневого логгера
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Минимальный уровень, который может быть обработан
root_logger.handlers = [console_handler, file_handler]
logger = logging.getLogger(__name__)

app = FastAPI()

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

# --- Класс: TextOnlyClassifier ---
class TextOnlyClassifier:
    def __init__(self):
        self.model: Optional[AdvancedTextClassifier] = None
        self.vectorizer = None
        self.threshold = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"TextOnlyClassifier initialized. Using device: {self.device}")

    def load(self, model_dir: str) -> bool:
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
            probabilities = torch.softmax(outputs, dim=1)
            prob_ad = probabilities[0][1].item()
        is_ad = prob_ad > self.threshold
        return prob_ad, is_ad

# --- Класс: MultimodalClassifier ---
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
            text_input_size = self.text_vectorizer.max_features; features_input_size = self.feature_scaler.n_features_in_
            self.model = MetaLearner(text_input_size=text_input_size, features_input_size=features_input_size).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False)); self.model.eval()
            logger.info(f"Multimodal model (MetaLearner) loaded from {model_path}.")
            self.image_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            return True
        except Exception as e:
            logger.error(f"Error loading multimodal model: {str(e)}", exc_info=True)
            self.model, self.text_vectorizer, self.feature_scaler = None, None, None
            return False

    def _extract_features(self, text: str, has_image: bool) -> np.ndarray:
        link_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        features = [len(text), link_count, text.count('@'), text.count('#'), text.count('/'), 0, 0, 0.0, 0.0, 1.0 if has_image else 0.0]
        return np.array(features).reshape(1, -1)

    def predict(self, text: str, image: Optional[Image.Image]) -> tuple[float, bool]:
        if not all([self.model, self.text_vectorizer, self.feature_scaler, self.image_transform]): raise RuntimeError("Multimodal model is not loaded.")
        text_vector = self.text_vectorizer.transform([text]).toarray().astype(np.float32)
        features_vector_raw = self._extract_features(text, has_image=(image is not None))
        features_vector = self.feature_scaler.transform(features_vector_raw).astype(np.float32)
        batch = {'text': torch.tensor(text_vector, dtype=torch.float32).to(self.device), 'features': torch.tensor(features_vector, dtype=torch.float32).to(self.device), 'labels': torch.tensor([0]).to(self.device)}
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

# --- Класс: ImageClassifier (для NSFW) ---
class ImageClassifier:
    def __init__(self):
        self.model: Optional[models.ResNet] = None
        self.transform: Optional[transforms.Compose] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = 0.5
        logger.info(f"ImageClassifier (NSFW) initialized. Using device: {self.device}")
    def load(self, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path): logger.error(f"Image model file not found: {model_path}"); return False
            self.model = models.resnet34(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3), nn.Linear(192, 1), nn.Sigmoid())
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.model.to(self.device); self.model.eval()
            self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            logger.info(f"Image model (ResNet34 for NSFW) loaded from {model_path}.")
            return True
        except Exception as e:
            logger.error(f"Error loading image model (NSFW): {str(e)}", exc_info=True)
            self.model, self.transform = None, None
            return False

# --- Инициализация всех классификаторов ---
multimodal_classifier = MultimodalClassifier()
text_only_classifier = TextOnlyClassifier()
nsfw_image_classifier = ImageClassifier()
multimodal_loaded = multimodal_classifier.load(model_dir=MODEL_MULTIMODAL_DIR)
if not multimodal_loaded: logger.warning("MULTIMODAL ad classification model FAILED to load.")
text_only_loaded = text_only_classifier.load(model_dir=MODEL_TEXT_ONLY_DIR)
if not text_only_loaded: logger.warning("TEXT-ONLY ad classification model FAILED to load.")
nsfw_model_loaded = nsfw_image_classifier.load(model_path=os.path.join(MODEL_RESNET_DIR, 'best_resnet_state_dict.pth'))
if not nsfw_model_loaded: logger.warning("NSFW classification model FAILED to load.")

# --- Модели Pydantic ---
class AdPredictionResponse(BaseModel): prediction_prob_ad: float; is_ad: bool; confidence: float; error: Optional[str] = None
class NsfwPredictionResponse(BaseModel): prediction_prob_nsfw: float; is_nsfw: bool; confidence: float; error: Optional[str] = None
class TextRequest(BaseModel): text: str

# --- API эндпоинты ---
@app.post("/api/classify_message/", response_model=AdPredictionResponse)
async def classify_message_endpoint(text: str = Form(""), image: Optional[Union[UploadFile, str]] = File(None)):
    if not multimodal_classifier.model: raise HTTPException(status_code=503, detail="Multimodal ad model not available.")
    try:
        pil_image, image_info = None, "no image"
        if isinstance(image, UploadFile) and image.filename:
            image_info = f"image present (filename: {image.filename})"
            contents = await image.read()
            if contents: pil_image = Image.open(io.BytesIO(contents))
        logger.info(f"Classifying multimodal. Text len: {len(text)}, {image_info}.")
        if not text.strip() and not pil_image: return AdPredictionResponse(prediction_prob_ad=0.0, is_ad=False, confidence=1.0, error="No content")
        prob_ad, is_ad = multimodal_classifier.predict(text=text, image=pil_image)
        logger.info(f"Multimodal result: is_ad={is_ad}, prob={prob_ad:.4f}")
        return AdPredictionResponse(prediction_prob_ad=prob_ad, is_ad=is_ad, confidence=prob_ad if is_ad else (1 - prob_ad))
    except Exception as e: logger.error(f"Error in multimodal endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify_text/", response_model=AdPredictionResponse)
async def classify_text_endpoint(request: TextRequest):
    if not text_only_classifier.model: raise HTTPException(status_code=503, detail="Text-only ad model not available.")
    try:
        if not request.text or not request.text.strip(): return AdPredictionResponse(prediction_prob_ad=0.0, is_ad=False, confidence=1.0, error="Empty text")
        logger.info(f"Classifying text-only. Text len: {len(request.text)}.")
        prob_ad, is_ad = text_only_classifier.predict(request.text)
        logger.info(f"Text-only result: is_ad={is_ad}, prob={prob_ad:.4f}")
        return AdPredictionResponse(prediction_prob_ad=prob_ad, is_ad=is_ad, confidence=prob_ad if is_ad else (1 - prob_ad))
    except Exception as e: logger.error(f"Error in text-only endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify_nsfw_image/", response_model=NsfwPredictionResponse)
async def classify_nsfw_image_endpoint(file: UploadFile = File(...)):
    if not nsfw_image_classifier.model: raise HTTPException(status_code=503, detail="NSFW model not available.")
    try:
        logger.info(f"Classifying NSFW. Filename: {file.filename}")
        contents = await file.read();
        if not contents: raise HTTPException(status_code=400, detail="Empty file")
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = nsfw_image_classifier.transform(img).unsqueeze(0).to(nsfw_image_classifier.device)
        with torch.no_grad(): prob_nsfw = nsfw_image_classifier.model(img_tensor).item()
        is_nsfw = prob_nsfw > nsfw_image_classifier.threshold
        logger.info(f"NSFW result: is_nsfw={is_nsfw}, prob={prob_nsfw:.4f}")
        return NsfwPredictionResponse(prediction_prob_nsfw=prob_nsfw, is_nsfw=is_nsfw, confidence=prob_nsfw if is_nsfw else (1 - prob_nsfw))
    except Exception as e: logger.error(f"Error in NSFW endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    logger.info("Health check requested.")
    return {"status": "healthy", "multimodal_model_loaded": multimodal_classifier.model is not None, "text_only_model_loaded": text_only_classifier.model is not None, "nsfw_model_loaded": nsfw_image_classifier.model is not None, "device": str(multimodal_classifier.device)}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_config=None)