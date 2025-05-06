from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import joblib
import uvicorn
import logging
import numpy as np
import cv2
from typing import Optional
import time
import os
import json

MODEL_DIR = 'result_torch_text'

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Класс для загрузки модели

class TextClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.threshold = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load(self, model_path, vectorizer_path, params_path, input_size):
        try:
            # Загрузка параметров модели из JSON
            with open(params_path, 'r') as f:
                params = json.load(f)

            num_layers = params.get("num_layers", 0)
            hidden_layers = [params[f"hidden_size_{i}"] for i in range(num_layers)]
            dropout = params.get("dropout", 0.5)
            activation = params.get("activation", "relu")
            use_batch_norm = params.get("use_batch_norm", False)

            # Загрузка векторизатора
            self.vectorizer = joblib.load(vectorizer_path)

            # Загрузка архитектуры
            from model_architecture import AdvancedTextClassifier
            self.model = AdvancedTextClassifier(
                input_size=input_size,
                hidden_layers=hidden_layers,
                num_classes=2,
                dropout=dropout,
                activation=activation,
                use_batch_norm=use_batch_norm
            ).to(self.device)

            # Загрузка весов
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

            logger.info("Модель успешно загружена")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}", exc_info=True)
            return False



text_classifier = TextClassifier()

try:
    text_classifier.load(
        model_path=os.path.join(MODEL_DIR, 'best_final_model.pth'),
        vectorizer_path=os.path.join(MODEL_DIR, 'final_vectorizer.pkl'),
        params_path=os.path.join(MODEL_DIR, 'final_best_params.json'),
        input_size=20000  # или другой, если отличается
    )
except Exception as e:
    logger.error(f"Ошибка инициализации модели: {e}")

app = FastAPI()

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s"
    )
    return response

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели запросов/ответов
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: float
    is_ad: bool
    confidence: float
    error: Optional[str] = None

# API endpoints
@app.post("/api/classify_text/", response_model=PredictionResponse)
async def classify_text(request: TextRequest):
    if not text_classifier.model:
        return PredictionResponse(
            prediction=0.0,
            is_ad=False,
            confidence=0.0,
            error="Текстовая модель не загружена"
        )
    
    try:
        # Векторизация текста
        vector = text_classifier.vectorizer.transform([request.text]).toarray()[0].astype(np.float32)
        tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(text_classifier.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = text_classifier.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            positive_prob = probabilities[0][1].item()
        
        # Формирование ответа
        return PredictionResponse(
            prediction=positive_prob,
            is_ad=positive_prob > text_classifier.threshold,
            confidence=positive_prob if positive_prob > text_classifier.threshold else 1 - positive_prob
        )
        
    except Exception as e:
        logger.error(f"Ошибка классификации: {e}", exc_info=True)
        return PredictionResponse(
            prediction=0.0,
            is_ad=False,
            confidence=0.0,
            error=str(e)
        )

# Health check
@app.get("/health")
async def health_check():
    return {
        "text_model_loaded": text_classifier.model is not None,
        "image_model_loaded": image_classifier is not None,
        "device": str(text_classifier.device)
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False
    )