import json
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
import nltk
import optuna
import joblib

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
    return clean_text(extract_text(message))

# Загрузка и обработка данных
def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)

    ad_texts = [extract_message_data(msg) for msg in ad_data['messages'] if clean_text(extract_text(msg))]
    non_ad_texts = [extract_message_data(msg) for msg in non_ad_data['messages'] if clean_text(extract_text(msg))]

    texts = ad_texts + non_ad_texts
    labels = [1] * len(ad_texts) + [0] * len(non_ad_texts)

    return texts, labels

# Класс для создания датасета
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # Токенизация текста с фиксированной длиной
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',  # Дополнение до max_len
            truncation=True,       # Усечение до max_len
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Функция для вычисления метрик
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Функция для оптимизации гиперпараметров
def optimize_bert(trial, model, train_dataset, test_dataset, data_collator):
    # Гиперпараметры для оптимизации
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    num_train_epochs = trial.suggest_int('num_train_epochs', 2, 5)

    # Настройка тренера
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        learning_rate=learning_rate
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Обучение модели
    trainer.train()

    # Оценка модели
    eval_result = trainer.evaluate()
    return eval_result['eval_accuracy']

# Основная функция
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ad_filepath = os.path.join(BASE_DIR, 'data/reklama', 'result.json')
    non_ad_filepath = os.path.join(BASE_DIR, 'data/nereklama', 'result.json')
    texts, labels = process_data(ad_filepath, non_ad_filepath)

    # Разделение данных на обучающую и тестовую выборки
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Загрузка токенизатора и модели BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Перенос модели на GPU, если доступно
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
        model = model.to('cuda')
    else:
        print("CUDA is not available. Using CPU.")

    # Создание датасетов
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=128)

    # Использование DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Оптимизация гиперпараметров
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_bert(trial, model, train_dataset, test_dataset, data_collator), n_trials=10)
    print("Best parameters:", study.best_params)

    # Финальное обучение модели с лучшими гиперпараметрами
    best_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    best_model.load_state_dict(model.state_dict())

    # Сохранение модели и токенизатора
    joblib.dump(best_model, 'bert_best_model.pkl')
    joblib.dump(tokenizer, 'bert_tokenizer.pkl')
    print('Best model and tokenizer saved to disk.')

if __name__ == "__main__":
    nltk.download('stopwords')
    main()