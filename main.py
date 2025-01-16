from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

# Загрузка модели и векторизатора
try:
    with open("randfor_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("randfor_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError as e:
    print(f"Ошибка при загрузке файлов: {e}")
    model, vectorizer = None, None

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/classify/")
async def classify_text(request: TextRequest):
    if model is None or vectorizer is None:
        return {"error": "Модель или векторизатор не загружены"}
    
    try:
        # Преобразуем текст с помощью векторизатора
        vector = vectorizer.transform([request.text])
        # Получаем предсказание вероятности
        prediction_prob = model.predict_proba(vector)[0][1]  # Предположим, что вторая колонка - это вероятность для "рекламы"
        
        # Классифицируем как рекламное, если вероятность больше 0.6
        is_ad = prediction_prob > 0.6
        
        return {
            "prediction": prediction_prob,
            "is_ad": is_ad
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
