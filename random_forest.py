import json
import csv
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import optuna
from datetime import datetime

# Загрузка данных
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Очистка текста
def clean_text(text):
    return text.replace('\n', ' ').replace('\r', '')

# Извлечение текста из сообщения
def extract_text(message):
    full_text = ""
    for part in message["text"]:
        if isinstance(part, dict):
            full_text += part["text"]
        else:
            full_text += part
    return full_text

# Извлечение данных из сообщения
def extract_message_data(message):
    return {
        "text": extract_text(message),
        "date": message.get("date", ""),
        "from": message.get("from", ""),
        "photo": message.get("photo", ""),
        "file_name": message.get("file_name", "")
    }

# Функция для сохранения данных в CSV
def save_to_csv(data, filename):
    with open(filename, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# Загрузка и обработка данных
def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)

    ad_texts = [extract_message_data(msg) for msg in ad_data['messages']]
    non_ad_texts = [extract_message_data(msg) for msg in non_ad_data['messages']]

    # Сохраняем посты в CSV
    save_to_csv(ad_texts + non_ad_texts, 'posts_data.csv')

    texts = [clean_text(msg['text']) for msg in ad_texts + non_ad_texts]
    labels = [1] * len(ad_texts) + [0] * len(non_ad_texts)

    return texts, labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ad_filepath = os.path.join(BASE_DIR, 'data/reklama', 'result.json')
non_ad_filepath = os.path.join(BASE_DIR, 'data/nereklama', 'result.json')
texts, labels = process_data(ad_filepath, non_ad_filepath)


# Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Векторизация текста с помощью TF-IDF
nltk.download('stopwords')
russian_stop_words = stopwords.words('russian')
vectorizer = TfidfVectorizer(max_features=10000, stop_words=russian_stop_words, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Балансировка классов с помощью SMOTE
smote = SMOTE(random_state=42)
train_vectors_balanced, train_labels_balanced = smote.fit_resample(train_vectors, train_labels)

# Оптимизация гиперпараметров Random Forest с помощью Optuna
def optimize_random_forest(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 10000)
    max_depth = trial.suggest_int('max_depth', 5, 1000)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 64)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_vectors_balanced, train_labels_balanced)
    predictions = model.predict(test_vectors)
    return accuracy_score(test_labels, predictions)

print('Optimize random forest')
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(optimize_random_forest, n_trials=1000)
print("Best Random Forest parameters:", study_rf.best_params)

# Финальное обучение модели с лучшими гиперпараметрами
rf_best = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
rf_best.fit(train_vectors_balanced, train_labels_balanced)
rf_predictions = rf_best.predict(test_vectors)

# Оценка модели
rf_accuracy = accuracy_score(test_labels, rf_predictions)
print(f"Random Forest Test Accuracy: {rf_accuracy}")

# Сохранение модели и векторизатора
joblib.dump(rf_best, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print('Best model and vectorizer saved to disk.')
