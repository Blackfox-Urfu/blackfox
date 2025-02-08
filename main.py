from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Загрузка модели и векторизатора через joblib
try:
    model = joblib.load("randfor_model.pkl")
    vectorizer = joblib.load("randfor_vectorizer.pkl")
except FileNotFoundError as e:
    print(f"Ошибка при загрузке файлов: {e}")
    model, vectorizer = None, None
except Exception as e:
    print(f"Общая ошибка при загрузке: {e}")
    model, vectorizer = None, None

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить доступ с любых доменов. Можно указать конкретные.
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все HTTP-методы.
    allow_headers=["*"],  # Разрешить все заголовки.
)

class TextRequest(BaseModel):
    text: str

def predict_ad_content(post):
    try:
        # Векторизация текста
        post_vector = vectorizer.transform([post])
        # Получение вероятности
        prediction_proba = model.predict_proba(post_vector)
        prediction = float(prediction_proba[0][1])  # Преобразование в float для сериализации
        logger.debug(f"Post: {post[:100]}... Prediction: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Ошибка в predict_ad_content: {e}", exc_info=True)
        raise e

@app.post("/classify/")
async def classify_text(request: TextRequest):
    if model is None or vectorizer is None:
        return {"error": "Модель или векторизатор не загружены"}
    
    try:
        prediction_prob = predict_ad_content(request.text)
        is_ad = bool(prediction_prob > 0.6)  # Преобразование в bool для сериализации
        return {
            "prediction": prediction_prob,
            "is_ad": is_ad
        }
    except Exception as e:
        logger.error(f"Ошибка в обработчике classify_text: {e}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)