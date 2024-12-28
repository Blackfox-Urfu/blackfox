import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
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

    ad_texts = [extract_message_data(msg) for msg in ad_data['messages'] if clean_text(extract_text(msg))]
    non_ad_texts = [extract_message_data(msg) for msg in non_ad_data['messages'] if clean_text(extract_text(msg))]

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
vectorizer = TfidfVectorizer(max_features=4000, stop_words=russian_stop_words, ngram_range=(1, 2))
# Извлекаем только тексты
train_texts_cleaned = [msg['text'] for msg in train_texts if 'text' in msg]
test_texts_cleaned = [msg['text'] for msg in test_texts if 'text' in msg]

train_vectors = vectorizer.fit_transform(train_texts_cleaned)
test_vectors = vectorizer.transform(test_texts_cleaned)


# Балансировка классов с помощью SMOTE (опционально)
smote = SMOTE(random_state=42)
train_vectors_balanced, train_labels_balanced = smote.fit_resample(train_vectors, train_labels)

# Оптимизация гиперпараметров CatBoost с Optuna
def optimize_catboost(trial):
    iterations = trial.suggest_int('iterations', 10, 600)
    depth = trial.suggest_int('depth', 4, 12)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10)
    loss_function = trial.suggest_categorical('loss_function', ['MultiClass', 'MultiClassOneVsAll'])

    
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=42,
        verbose=0,
        loss_function=loss_function,
        boosting_type='Plain',
        task_type="GPU",
        gpu_cat_features_storage='CpuPinnedMemory'
    )
    model.fit(train_vectors_balanced, train_labels_balanced)
    predictions = model.predict(test_vectors)
    return accuracy_score(test_labels, predictions)

print('Optimize CatBoost')
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(optimize_catboost, n_trials=100)
print("Best CatBoost parameters:", study_cb.best_params)

# Финальное обучение CatBoost с лучшими параметрами
cb_best = CatBoostClassifier(**study_cb.best_params, task_type='GPU', devices='0', random_seed=42, verbose=0)
cb_best.fit(train_vectors_balanced, train_labels_balanced)

# Оценка модели
cb_predictions = cb_best.predict(test_vectors)
cb_accuracy = accuracy_score(test_labels, cb_predictions)
print(f"CatBoost Test Accuracy: {cb_accuracy}")

# Сохранение модели и векторизатора
joblib.dump(cb_best, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print('Best CatBoost model and vectorizer saved to disk.')
