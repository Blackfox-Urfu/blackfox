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

# --- Настройка путей и импортов ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REKLAMA_MODELS_DIR = os.path.join(PROJECT_ROOT, "app", "learn", "reklama_classification_models")
if REKLAMA_MODELS_DIR not in sys.path:
    sys.path.insert(0, REKLAMA_MODELS_DIR)
MODEL_MULTIMODAL_DIR = os.path.join(PROJECT_ROOT, "model", "multimodal")
MODEL_RESNET_DIR = os.path.join(PROJECT_ROOT, "model", "resnet")
try:
    from torch_models import MetaLearner
except ImportError:
    print(f"CRITICAL: Could not import MetaLearner from {REKLAMA_MODELS_DIR}")
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

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.addFilter(RequestIdFilter())

file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight", interval=1, backupCount=7, encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.addFilter(RequestIdFilter())

logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logger = logging.getLogger(__name__)

app = FastAPI()

# Middleware для установки/очистки request_id
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4().hex[:8])
    thread_local.request_id = request_id
    
    logger.info(f"Request started: {request.method} {request.url.path} from {request.client.host}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Request finished: {response.status_code} in {process_time:.4f}s")
    
    # Сбрасываем id, чтобы не переиспользовать его случайно
    setattr(thread_local, 'request_id', None)
    return response

# --- Класс: MultimodalClassifier (для рекламы) ---
class MultimodalClassifier:
    def __init__(self):
        self.model: Optional[MetaLearner] = None
        self.text_vectorizer = None
        self.feature_scaler = None
        self.image_transform: Optional[transforms.Compose] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"MultimodalClassifier initialized. Using device: {self.device}")

    def load(self, model_dir: str) -> bool:
        model_path = os.path.join(model_dir, 'best_model.pth')
        vectorizer_path = os.path.join(model_dir, 'text_vectorizer.pkl')
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')

        try:
            if not all(os.path.exists(p) for p in [model_path, vectorizer_path, scaler_path]):
                logger.error("One or more multimodal model files not found!")
                return False
            self.text_vectorizer = joblib.load(vectorizer_path)
            self.feature_scaler = joblib.load(scaler_path)
            logger.info("Text vectorizer and feature scaler loaded.")
            text_input_size = self.text_vectorizer.max_features
            features_input_size = self.feature_scaler.n_features_in_
            self.model = MetaLearner(text_input_size=text_input_size, features_input_size=features_input_size).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.model.eval()
            logger.info(f"Multimodal model (MetaLearner) loaded from {model_path}.")
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
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
        if not all([self.model, self.text_vectorizer, self.feature_scaler, self.image_transform]):
             raise RuntimeError("Multimodal model is not properly loaded.")
        text_vector = self.text_vectorizer.transform([text]).toarray().astype(np.float32)
        features_vector_raw = self._extract_features(text, has_image=(image is not None))
        features_vector = self.feature_scaler.transform(features_vector_raw).astype(np.float32)
        batch = {
            'text': torch.tensor(text_vector, dtype=torch.float32).to(self.device),
            'features': torch.tensor(features_vector, dtype=torch.float32).to(self.device),
            'labels': torch.tensor([0]).to(self.device)
        }
        if image:
            try:
                image_tensor = self.image_transform(image.convert('RGB')).unsqueeze(0)
                batch['images'] = image_tensor.to(self.device)
                batch['image_indices'] = torch.tensor([0], dtype=torch.long).to(self.device)
            except Exception as e:
                logger.warning(f"Could not process image, proceeding with text/features only. Error: {e}")
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            prob_ad = probabilities[0][1].item()
        is_ad = prob_ad > 0.5
        return prob_ad, is_ad

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
            if not os.path.exists(model_path):
                logger.error(f"Image model file not found: {model_path}")
                return False
            self.model = models.resnet34(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3), nn.Linear(192, 1), nn.Sigmoid())
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.model.to(self.device)
            self.model.eval()
            self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
            logger.info(f"Image model (ResNet34 for NSFW) loaded from {model_path}.")
            return True
        except Exception as e:
            logger.error(f"Error loading image model (NSFW): {str(e)}", exc_info=True)
            self.model, self.transform = None, None
            return False

# --- Инициализация и приложение FastAPI ---
multimodal_classifier = MultimodalClassifier()
nsfw_image_classifier = ImageClassifier()
multimodal_loaded = multimodal_classifier.load(model_dir=MODEL_MULTIMODAL_DIR)
if not multimodal_loaded: logger.warning("ADVERTISEMENT classification model FAILED to load.")
nsfw_model_loaded = nsfw_image_classifier.load(model_path=os.path.join(MODEL_RESNET_DIR, 'best_resnet_state_dict.pth'))
if not nsfw_model_loaded: logger.warning("NSFW classification model FAILED to load.")


# --- Модели Pydantic ---
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

# --- API эндпоинты ---
@app.post("/api/classify_message/", response_model=AdPredictionResponse)
async def classify_message_endpoint(text: str = Form(""), image: Optional[Union[UploadFile, str]] = File(None)):
    if not multimodal_classifier.model:
        logger.error("Ad classification model is not available.")
        raise HTTPException(status_code=503, detail="Ad classification model is not available.")
    try:
        pil_image = None
        image_info = "no image"
        if isinstance(image, UploadFile) and image.filename:
            image_info = f"image present (filename: {image.filename})"
            contents = await image.read()
            if contents:
                try:
                    pil_image = Image.open(io.BytesIO(contents))
                except (IOError, UnidentifiedImageError) as e:
                    logger.warning(f"Could not open uploaded file {image.filename} as image, ignoring. Error: {e}", exc_info=True)
            else:
                logger.warning(f"Uploaded file {image.filename} is empty, ignoring.")
        
        logger.info(f"Classifying message. Text length: {len(text)}, {image_info}.")
        if not text.strip() and not pil_image:
            return AdPredictionResponse(prediction_prob_ad=0.0, is_ad=False, confidence=1.0, error="No content provided (text and image are empty).")
        
        prob_ad, is_ad = multimodal_classifier.predict(text=text, image=pil_image)
        logger.info(f"Ad classification result: is_ad={is_ad}, probability={prob_ad:.4f}")
        return AdPredictionResponse(prediction_prob_ad=prob_ad, is_ad=is_ad, confidence=prob_ad if is_ad else (1 - prob_ad))
    except Exception as e:
        logger.error(f"Error during multimodal classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during classification: {str(e)}")

@app.post("/api/classify_nsfw_image/", response_model=NsfwPredictionResponse)
async def classify_nsfw_image_endpoint(file: UploadFile = File(...)):
    if not nsfw_image_classifier.model:
        logger.error("NSFW classification model is not available.")
        raise HTTPException(status_code=503, detail="NSFW classification model is not available.")
    try:
        logger.info(f"Classifying NSFW image. Filename: {file.filename}, content-type: {file.content_type}")
        contents = await file.read()
        if not contents: raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = nsfw_image_classifier.transform(img).unsqueeze(0).to(nsfw_image_classifier.device)
        with torch.no_grad():
            prob_nsfw = nsfw_image_classifier.model(img_tensor).item()
        is_nsfw_prediction = prob_nsfw > nsfw_image_classifier.threshold
        logger.info(f"NSFW classification result: is_nsfw={is_nsfw_prediction}, probability={prob_nsfw:.4f}")
        return NsfwPredictionResponse(prediction_prob_nsfw=prob_nsfw, is_nsfw=is_nsfw_prediction, confidence=prob_nsfw if is_nsfw_prediction else (1 - prob_nsfw))
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error processing NSFW image {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during NSFW image processing: {str(e)}")

@app.get("/health")
async def health_check():
    logger.info("Health check requested.")
    return {"status": "healthy", "ad_model_loaded": multimodal_classifier.model is not None, "ad_model_device": str(multimodal_classifier.device), "nsfw_model_loaded": nsfw_image_classifier.model is not None, "nsfw_model_device": str(nsfw_image_classifier.device),}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_config=None)