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

            # weights_only=False is needed if the checkpoint contains more than just weights (e.g., optimizer state, config)
            # If you are sure it only contains model_state and config, True might be safer against pickle exploits.
            # Given your previous structure, False is likely correct.
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

# --- Класс ImageClassifier (ИСПРАВЛЕННЫЙ) ---
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

            # 1. Initialize ResNet34 with pre-trained weights for the base
            # Use the new 'weights' API
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            num_ftrs = self.model.fc.in_features # This will be 512 for resnet34

            # 2. Define the FC layer structure that matches the checkpoint
            # Based on the error messages:
            # fc.0.weight: checkpoint [832, 512], current model [1024, 512] -> checkpoint has Linear(512, 832)
            # fc.4.weight: checkpoint [576, 832], current model [512, 1024] -> checkpoint has Linear(832, 576)
            # fc.8.weight: checkpoint [1, 576], current model [1, 512]   -> checkpoint has Linear(576, 1)
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 832),
                nn.BatchNorm1d(832),
                nn.ReLU(),
                nn.Dropout(0.5), # Assuming this was the original dropout, adjust if known
                nn.Linear(832, 576),
                nn.BatchNorm1d(576),
                nn.ReLU(),
                nn.Dropout(0.3), # Assuming this was the original dropout, adjust if known
                nn.Linear(576, 1),
                nn.Sigmoid()
            )

            # Load the state_dict from the checkpoint.
            checkpoint_state_dict = torch.load(model_path, map_location=self.device)
            
            # Optional: Log keys to debug if issues persist
            # logger.info(f"Keys in loaded checkpoint state_dict for image model: {list(checkpoint_state_dict.keys())}")
            # logger.info(f"Keys in current model state_dict before loading: {list(self.model.state_dict().keys())}")

            # 3. Load with strict=False.
            # This will load the weights for the layers that match (i.e., your fc layer)
            # and ignore the missing keys (the ResNet base layers, which are already pre-trained).
            # It will also ignore unexpected keys if any.
            # If there are still size mismatches for existing keys (e.g. fc.0.weight), it will error.
            incompatible_keys = self.model.load_state_dict(checkpoint_state_dict, strict=False)
            
            if incompatible_keys.missing_keys:
                logger.warning(f"Missing keys when loading image model state_dict: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                logger.warning(f"Unexpected keys when loading image model state_dict: {incompatible_keys.unexpected_keys}")

            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info(f"Image model (ResNet34) loaded. Base uses ImageNet weights. FC layer loaded from {model_path}.")
            return True
        except RuntimeError as e: # Catch specific RuntimeError for state_dict issues
            logger.error(f"RuntimeError loading image model state_dict: {str(e)}", exc_info=True)
            # Log keys from model and checkpoint for detailed comparison if a RuntimeError occurs
            try:
                if self.model:
                    logger.error(f"Current model keys: {list(self.model.state_dict().keys())}")
                if os.path.exists(model_path):
                    checkpoint_for_debug = torch.load(model_path, map_location='cpu') # Load to CPU for inspection
                    logger.error(f"Checkpoint keys: {list(checkpoint_for_debug.keys())}")
            except Exception as debug_e:
                logger.error(f"Error during debug logging of keys: {debug_e}")
            self.model = None
            self.transform = None
            return False
        except Exception as e:
            logger.error(f"Generic error loading image model: {str(e)}", exc_info=True)
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
# Обернем этот middleware в другой вызов add_middleware, чтобы он шел до основного логгера запросов
# Это чисто для порядка, чтобы лог о слишком большом файле появлялся до лога об обработке запроса
async def check_file_size_middleware(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/api/classify_image/":
        content_length_header = request.headers.get("content-length")
        if content_length_header:
            try:
                content_length = int(content_length_header)
                MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
                
                if content_length > MAX_FILE_SIZE:
                    logger.warning(f"Attempt to upload too large file: {content_length} bytes. Max: {MAX_FILE_SIZE} bytes for {request.url.path}")
                    # Возвращаем HTTPException напрямую, FastAPI обработает его.
                    # Нет необходимости создавать Response вручную для этого.
                    raise HTTPException(
                        status_code=413, # Payload Too Large
                        detail=f"Файл слишком большой. Максимальный размер: {MAX_FILE_SIZE/(1024*1024):.0f}MB."
                    )
            except ValueError:
                logger.warning(f"Invalid content-length header: {content_length_header} for {request.url.path}")
                raise HTTPException(status_code=400, detail="Invalid content-length header.")
    
    return await call_next(request)

# Добавляем middleware в правильном порядке
app.middleware("http")(check_file_size_middleware) # Сначала проверка размера

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

class TextPredictionResponse(BaseModel):
    prediction_prob_ad: float
    is_ad: bool
    confidence: float
    error: Optional[str] = None

class ImagePredictionResponse(BaseModel):
    prediction_prob_nsfw: float
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

@app.post("/api/classify_image/", response_model=ImagePredictionResponse)
async def classify_image_endpoint(file: UploadFile = File(...)):
    if not image_classifier.model or not image_classifier.transform:
        logger.error("Image model not loaded for /api/classify_image/")
        raise HTTPException(status_code=503, detail="Image classification model is not available.")

    try:
        logger.debug(f"Processing image: {file.filename}, content_type: {file.content_type}")
        contents = await file.read()
        
        if not contents:
            logger.warning(f"Uploaded file {file.filename} is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as pil_e:
            logger.error(f"Error opening image {file.filename} with PIL: {pil_e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Could not process image file. Ensure it's a valid image format. Error: {str(pil_e)}")

        img_tensor = image_classifier.transform(img).unsqueeze(0).to(image_classifier.device)
        
        with torch.no_grad():
            prob_nsfw = image_classifier.model(img_tensor).item() 
        
        is_nsfw_prediction = prob_nsfw > image_classifier.threshold
        
        logger.info(f"Image classification result for {file.filename}: prob_nsfw={prob_nsfw:.4f}, is_nsfw={is_nsfw_prediction}")
        
        return ImagePredictionResponse(
            prediction_prob_nsfw=prob_nsfw,
            is_nsfw=is_nsfw_prediction,
            confidence=prob_nsfw if is_nsfw_prediction else (1 - prob_nsfw)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during image processing: {str(e)}")

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
    # Ensure MODEL_FILES_RESNET_DIR and MODEL_FILES_TEXT_DIR are correct before Uvicorn starts
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"MODEL_FILES_TEXT_DIR: {MODEL_FILES_TEXT_DIR}")
    logger.info(f"MODEL_FILES_RESNET_DIR: {MODEL_FILES_RESNET_DIR}")

    # Check if model files exist before attempting to load
    text_model_file = os.path.join(MODEL_FILES_TEXT_DIR, 'best_final_model.pth')
    vectorizer_file = os.path.join(MODEL_FILES_TEXT_DIR, 'final_vectorizer.pkl')
    image_model_file = os.path.join(MODEL_FILES_RESNET_DIR, 'best_resnet.pth')

    if not os.path.exists(text_model_file):
        logger.critical(f"Critical: Text model file not found at {text_model_file}. Server may not function correctly.")
    if not os.path.exists(vectorizer_file):
        logger.critical(f"Critical: Vectorizer file not found at {vectorizer_file}. Server may not function correctly.")
    if not os.path.exists(image_model_file):
        logger.critical(f"Critical: Image model file not found at {image_model_file}. Server may not function correctly.")

    uvicorn.run(
        "main:app", # refers to this file (main.py) and the app instance
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True # Uvicorn's access log can be useful, even with custom logging.
                        # Set to False if you only want your custom log_requests middleware.
    )