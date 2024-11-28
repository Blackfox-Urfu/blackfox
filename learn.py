import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
import optuna

# Проверка доступности GPU
print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# --- 1. Загрузка и обработка данных ---
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', '')

def extract_text(message):
    full_text = ""
    for part in message["text"]:
        full_text += part["text"] if isinstance(part, dict) else part
    return full_text

def extract_message_data(message):
    return {
        "id": message["id"],
        "date": message["date"],
        "from": message.get("from"),
        "author": message.get("author"),
        "forwarded_from": message.get("forwarded_from"),
        "photo": message.get("photo"),
        "text": extract_text(message)
    }

def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)

    ad_texts = [clean_text(extract_message_data(msg)['text']) for msg in ad_data['messages']]
    non_ad_texts = [clean_text(extract_message_data(msg)['text']) for msg in non_ad_data['messages']]

    texts = ad_texts + non_ad_texts
    labels = [1] * len(ad_texts) + [0] * len(non_ad_texts)
    return texts, labels

# --- 2. Пути к файлам и обработка текста ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ad_filepath = os.path.join(BASE_DIR, 'data/reklama', 'result.json')
non_ad_filepath = os.path.join(BASE_DIR, 'data/nereklama', 'result.json')

texts, labels = process_data(ad_filepath, non_ad_filepath)

# Разделение данных
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Токенизация
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# --- 3. Создание модели ---
def create_model(trial, max_length):
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.1)
    dense_units_1 = trial.suggest_int("dense_units_1", 64, 256, step=16)
    dense_units_2 = trial.suggest_int("dense_units_2", 32, 128, step=16)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = Sequential([
        Dense(dense_units_1, activation='relu', input_shape=(max_length,)),
        Dropout(dropout_rate),
        Dense(dense_units_2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 4. Оптимизация гиперпараметров с Optuna ---
def objective(trial):
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    validation_split = trial.suggest_float("validation_split", 0.1, 0.3, step=0.05)
    epochs = trial.suggest_int("epochs", 5, 20)

    class_weight = {
        0: len(train_labels) / (2 * np.sum(train_labels == 0)),
        1: len(train_labels) / (2 * np.sum(train_labels == 1))
    }

    model = create_model(trial, max_length)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_padded, train_labels,
        epochs=epochs, batch_size=batch_size,
        validation_split=validation_split,
        class_weight=class_weight,
        callbacks=[early_stopping], verbose=0
    )
    _, accuracy = model.evaluate(test_padded, test_labels, verbose=0)
    return accuracy

# Указываем, что результаты должны сохраняться в базе данных SQLite
study = optuna.create_study(direction="maximize", storage="sqlite:///optuna_study.db", load_if_exists=True)
study.optimize(objective, n_trials=300)

# --- 5. Обучение и оценка модели ---
best_params = study.best_params
print(f"Best Parameters: {best_params}")

final_model = create_model(study.best_trial, max_length)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

final_model.fit(
    train_padded, train_labels,
    epochs=10, batch_size=best_params['batch_size'],
    validation_split=0.2, class_weight={0: len(train_labels) / (2 * np.sum(train_labels == 0)),
                                        1: len(train_labels) / (2 * np.sum(train_labels == 1))},
    callbacks=[early_stopping], verbose=1
)

_, final_accuracy = final_model.evaluate(test_padded, test_labels)
print(f"Final Accuracy: {final_accuracy}")
