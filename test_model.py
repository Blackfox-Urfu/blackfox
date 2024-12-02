import csv
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Увеличиваем лимит размера поля CSV
csv.field_size_limit(sys.maxsize)

# Параметры
MAX_SEQUENCE_LENGTH = 100

# Загрузка модели
model_path = "best_model.keras"
print(f"Loading model from {model_path}...")
model = load_model(model_path)
print("Model loaded successfully.")

# Загрузка токенизатора
print("Loading tokenizer from tokenizer.pkl...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded successfully.")

# Построчная обработка текста
print("\n--- Predictions ---")
input_file = "extracted_texts.csv"
with open(input_file, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Пропустить заголовок

    for row in reader:
        if row:
            text = row[0]
            # Преобразование текста в последовательности
            sequence = tokenizer.texts_to_sequences([text])
            data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
            # Предсказание
            prediction = model.predict(data, batch_size=1)
            predicted_class = (prediction > 0.5).astype(int)[0][0]

            # Вывод результата
            print(f"Text: {text}... | Prediction: {prediction}")

print("\nProcessing completed.")
