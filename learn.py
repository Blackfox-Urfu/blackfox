import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import joblib
from datetime import datetime
import os

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
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Балансировка классов с помощью SMOTE
smote = SMOTE(random_state=42)
train_vectors_balanced, train_labels_balanced = smote.fit_resample(train_vectors, train_labels)

# Определение моделей для тестирования
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),  # Увеличение max_iter
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Гиперпараметрическая оптимизация для Random Forest и Gradient Boosting
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],  # Регуляризация
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5]
}

# Грид-поиск для моделей
# Грид-поиск для моделей с многопоточностью
grid_searches = {
    'Logistic Regression': GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1),
    'Random Forest': GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1),
    'Gradient Boosting': GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
}


# Обучение и оценка моделей
results = {}
for model_name, grid_search in grid_searches.items():
    grid_search.fit(train_vectors_balanced, train_labels_balanced)
    best_model = grid_search.best_estimator_

    predictions = best_model.predict(test_vectors)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)

    results[model_name] = {
        'accuracy': accuracy,
        'report': report,
        'conf_matrix': conf_matrix
    }

    print(f'{model_name} Test Accuracy: {accuracy}')
    print(f'{model_name} Classification Report:')
    print(report)

    

# Сохранение лучшей модели и векторизатора
best_model_name = max(results, key=lambda name: results[name]['accuracy'])
best_model = grid_searches[best_model_name].best_estimator_

joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print('Best model and vectorizer saved to disk.')
