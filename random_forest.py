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
from collections import Counter
import numpy as np

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

# Балансировка классов с помощью SMOTE
smote = SMOTE(random_state=42)
train_vectors_balanced, train_labels_balanced = smote.fit_resample(train_vectors, train_labels)

# Подсчет количества сообщений в каждом классе после SMOTE
balanced_class_counts = Counter(train_labels_balanced)
# Вывод количества сообщений для каждого класса после SMOTE
print(f'{'-'*10}\nSMOTE DATA')
print("Balanced dataset class distribution:")
for label, count in balanced_class_counts.items():
    print(f"Class {label}: {count}")
print(f'{'-'*10}\n')

# Расчет Class imbalance ratio после SMOTE
class_imbalance_ratio = balanced_class_counts[0] / balanced_class_counts[1]
print(f"Class imbalance ratio after SMOTE: {class_imbalance_ratio:.2f}")

# Оптимизация гиперпараметров Random Forest с помощью Optuna
def optimize_random_forest(trial):
    n_estimators = trial.suggest_int('n_estimators', 70, 580)
    max_depth = trial.suggest_int('max_depth', 100, 685)
    min_samples_split = trial.suggest_int('min_samples_split', 2,32)
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
study_rf.optimize(optimize_random_forest, n_trials=100)
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
