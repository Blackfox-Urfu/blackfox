import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import os
from nltk.corpus import stopwords
import nltk
import optuna

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
        "id": message["id"],
        "date": message["date"],
        "from": message.get("from", None),
        "author": message.get("author", None),
        "forwarded_from": message.get("forwarded_from", None),
        "photo": message.get("photo", None),
        
        "text": extract_text(message)
    }

# Загрузка и обработка данных
def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)

    ad_texts = [clean_text(extract_message_data(msg)['text']) for msg in ad_data['messages']]
    non_ad_texts = [clean_text(extract_message_data(msg)['text']) for msg in non_ad_data['messages']]

    texts = ad_texts + non_ad_texts
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

# Оптимизация гиперпараметров с Optuna
def optimize_random_forest(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
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

def optimize_gradient_boosting(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(train_vectors_balanced, train_labels_balanced)
    predictions = model.predict(test_vectors)
    return accuracy_score(test_labels, predictions)

# Оптимизация Random Forest
print('Optimize random forest')
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(optimize_random_forest, n_trials=100)
print("Best Random Forest parameters for RANDOM FOREST:", study_rf.best_params)

# Оптимизация Gradient Boosting
print('Optimize gradient boosting')
study_gb = optuna.create_study(direction='maximize')
study_gb.optimize(optimize_gradient_boosting, n_trials=100)
print("Best Gradient Boosting parameters:", study_gb.best_params)

# Финальное обучение моделей с лучшими гиперпараметрами
rf_best = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
rf_best.fit(train_vectors_balanced, train_labels_balanced)
rf_predictions = rf_best.predict(test_vectors)

gb_best = GradientBoostingClassifier(**study_gb.best_params, random_state=42)
gb_best.fit(train_vectors_balanced, train_labels_balanced)
gb_predictions = gb_best.predict(test_vectors)

# Оценка моделей
rf_accuracy = accuracy_score(test_labels, rf_predictions)
gb_accuracy = accuracy_score(test_labels, gb_predictions)
print(f"Random Forest Test Accuracy: {rf_accuracy}")
print(f"Gradient Boosting Test Accuracy: {gb_accuracy}")

# Сохранение лучшей модели и векторизатора
best_model = rf_best if rf_accuracy > gb_accuracy else gb_best
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print('Best model and vectorizer saved to disk.')