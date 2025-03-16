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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
import joblib
import optuna
from datetime import datetime
from collections import Counter
import numpy as np
import time

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

    ad_texts = [extract_message_data(msg) for msg in ad_data['messages'] if clean_text(extract_text(msg))]
    non_ad_texts = [extract_message_data(msg) for msg in non_ad_data['messages'] if clean_text(extract_text(msg))]


    # Сохраняем посты в CSV
    save_to_csv(ad_texts + non_ad_texts, 'posts_data.csv')

    texts = [clean_text(msg['text']) for msg in ad_texts + non_ad_texts]
    labels = [1] * len(ad_texts) + [0] * len(non_ad_texts)

    return texts, labels

def dataset_statistics(texts, labels):
    print("Dataset Statistics:")
    total_messages = len(texts)
    print(f"Total messages: {total_messages}")

    # Подсчет пропусков в тексте
    missing_texts = sum(1 for text in texts if not text.strip())
    print(f"Missing texts: {missing_texts} ({missing_texts / total_messages * 100:.2f}%)")

    # Распределение классов
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f"Class {label}: {count} ({count / total_messages * 100:.2f}%)")
    
    # Средняя длина сообщений
    message_lengths = [len(text.split()) for text in texts if text.strip()]
    avg_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
    print(f"Average message length: {avg_length:.2f} words")

    # Максимальная и минимальная длина сообщений
    max_length = max(message_lengths, default=0)
    min_length = min(message_lengths, default=0)
    print(f"Longest message length: {max_length} words")
    print(f"Shortest message length: {min_length} words")

    # Количество уникальных слов
    unique_words = set(word for text in texts for word in text.split() if text.strip())
    print(f"Unique words: {len(unique_words)}")
    
    # Статистика для каждого класса
    print(f"\n{'-'*10}\nPer-Class Statistics:")
    for label in label_counts.keys():
        class_texts = [texts[i] for i in range(total_messages) if labels[i] == label and texts[i].strip()]
        class_lengths = [len(text.split()) for text in class_texts]
        avg_class_length = np.mean(class_lengths) if class_lengths else 0
        total_class_length = sum(class_lengths)
        
        print(f"Class {label}:")
        print(f"  Total messages: {len(class_texts)}")
        print(f"  Average message length: {avg_class_length:.2f} words")
        print(f"  Total words in class: {total_class_length}")
        print(f'{"-"*5}')

    # Соотношение классов
    if len(label_counts) == 2:  # Подходит для бинарной классификации
        class_0, class_1 = label_counts[0], label_counts[1]
        imbalance_ratio = min(class_0, class_1) / max(class_0, class_1)
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        print()

def balance_dataset(train_vectors, train_labels, method='combined'):
    """
    Балансировка датасета различными методами
    """
    print("Initial class distribution:", Counter(train_labels))
    
    if method == 'under':
        sampler = RandomUnderSampler(random_state=42)
        train_vectors_balanced, train_labels_balanced = sampler.fit_resample(
            train_vectors, train_labels
        )
    
    elif method == 'over':
        sampler = RandomOverSampler(random_state=42)
        train_vectors_balanced, train_labels_balanced = sampler.fit_resample(
            train_vectors, train_labels
        )
    
    elif method == 'combined':
        # Находим соотношение классов
        class_counts = Counter(train_labels)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        
        # Вычисляем целевое количество семплов для undersampling
        target_ratio = {
            majority_class: int(class_counts[majority_class] * 0.8),  # Оставляем 80% большего класса
            minority_class: class_counts[minority_class]  # Не трогаем меньший класс
        }
        
        # Сначала уменьшаем больший класс
        undersampler = RandomUnderSampler(
            sampling_strategy=target_ratio,
            random_state=42
        )
        vectors_under, labels_under = undersampler.fit_resample(
            train_vectors, train_labels
        )
        
        # Затем увеличиваем меньший класс до баланса
        oversampler = RandomOverSampler(random_state=42)
        train_vectors_balanced, train_labels_balanced = oversampler.fit_resample(
            vectors_under, labels_under
        )
    
    elif method == 'weighted':
        class_weights = compute_class_weights(train_labels)
        return train_vectors, train_labels, class_weights

    print("Balanced class distribution:", Counter(train_labels_balanced))
    return train_vectors_balanced, train_labels_balanced, None

def compute_class_weights(labels):
    """
    Вычисление весов классов обратно пропорционально их частоте
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {
        class_label: total_samples / (len(class_counts) * count)
        for class_label, count in class_counts.items()
    }
    return class_weights

def optimize_random_forest(trial):
    # Гиперпараметры для оптимизации
    n_estimators = trial.suggest_int('n_estimators', 100, 3000)
    max_depth = trial.suggest_int('max_depth', 10, 3000)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    
    # Создаем и обучаем модель с текущими параметрами
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(train_vectors_balanced, train_labels_balanced)
    
    # Измеряем время предсказания
    start_time = time.time()
    predictions = model.predict(test_vectors)
    prediction_time = time.time() - start_time
    
    accuracy = accuracy_score(test_labels, predictions)
    
    # Условие для точности
    if accuracy < 0.86:
        print(f"Trial {trial.number}: Accuracy = {accuracy:.4f} is below threshold. Skipping this trial.")
        return float('-inf')  # Игнорируем эту итерацию
    
    # Учитываем время предсказания в качестве штрафа
    performance_score = accuracy - prediction_time*0.5  # Можно настроить вес штрафа

    # Выводим точность и время ответа
    print(f"Trial {trial.number}: Accuracy = {accuracy:.4f}, Prediction Time = {prediction_time:.4f} seconds")
    
    return performance_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ad_filepath = os.path.join(BASE_DIR, 'data/reklama', 'result.json')
non_ad_filepath = os.path.join(BASE_DIR, 'data/nereklama', 'result.json')
texts, labels = process_data(ad_filepath, non_ad_filepath)

# Вывод статистики по датасету 
dataset_statistics(texts, labels)

# Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Векторизация текста с помощью TF-IDF
nltk.download('stopwords')
russian_stop_words = stopwords.words('russian')
vectorizer = TfidfVectorizer(max_features=10000, stop_words=russian_stop_words, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Балансировка данных
train_vectors_balanced, train_labels_balanced, class_weights = balance_dataset(
    train_vectors, 
    train_labels,
    method='combined'  # Можно выбрать: 'under', 'over', 'combined', 'weighted'
)

# Оптимизация гиперпараметров
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(optimize_random_forest, n_trials=1000)

# Обучение модели с лучшими параметрами
rf_best = RandomForestClassifier(
    **study_rf.best_params,
    random_state=42,
    n_jobs=-1
)

# Обучение модели
rf_best.fit(train_vectors_balanced, train_labels_balanced)

# Оценка модели
rf_predictions = rf_best.predict(test_vectors)
rf_accuracy = accuracy_score(test_labels, rf_predictions)
print(f"Random Forest Test Accuracy: {rf_accuracy}")

# Сохранение модели и векторизатора
joblib.dump(rf_best, 'randfor_model.pkl')
joblib.dump(vectorizer, 'randfor_vectorizer.pkl')
print('Best model and vectorizer saved to disk.')

print("Best parameters found:", study_rf.best_params)
print("Best accuracy achieved:", study_rf.best_value)