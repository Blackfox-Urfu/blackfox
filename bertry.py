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
    # Гиперпараметры
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 5e-4, log=True)
    num_train_epochs = trial.suggest_int('num_train_epochs', 2, 10)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    warmup_steps = trial.suggest_int('warmup_steps', 0, 1000)
    batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])
    eval_batch_size = trial.suggest_categorical('per_device_eval_batch_size', [16, 32, 64])
    optimizer = trial.suggest_categorical('optimizer', ['adamw_hf', 'adamw_torch', 'sgd', 'adafactor'])
    scheduler_type = trial.suggest_categorical('scheduler_type', ['linear', 'cosine', 'cosine_with_restarts'])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 8)
    hidden_dropout_prob = trial.suggest_float('hidden_dropout_prob', 0.1, 0.3)
    attention_probs_dropout_prob = trial.suggest_float('attention_probs_dropout_prob', 0.1, 0.3)
    freeze_layers = trial.suggest_int('freeze_layers', 0, 9)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        optim=optimizer,
        lr_scheduler_type=scheduler_type,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Заморозка первых слоев модели
    for param in model.bert.encoder.layer[:freeze_layers]:
        for p in param.parameters():
            p.requires_grad = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    
    # Возвращаем значение accuracy
    return metrics['eval_accuracy']


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
    study.optimize(lambda trial: optimize_bert(trial, model, train_dataset, test_dataset, data_collator), n_trials=60)
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