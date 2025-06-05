import json
import csv
import os
import time
from datetime import datetime
from collections import Counter

import nltk
# --- Исправлен импорт stopwords ---
from nltk.corpus import stopwords
# --- Конец исправления ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import optuna
from optuna.samplers import TPESampler
import joblib

# Игнорируем предупреждения
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- НАЧАЛО ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---
# Предполагаем, что этот скрипт находится в app/learn/torch_text/torch_text.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Поднимаемся на три уровня вверх до PROJECT_ROOT: torch_text -> learn -> app -> PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# Обновленная директория для результатов, теперь внутри model/torch_text/
RESULTS_DIR = os.path.join(PROJECT_ROOT, "model", "torch_text")
os.makedirs(RESULTS_DIR, exist_ok=True)
# --- КОНЕЦ ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---

# --- Определяем num_workers ---
try:
    num_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 1
except NotImplementedError:
    num_workers = 4
print(f"Using num_workers = {num_workers} for DataLoaders.")

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
print(f"Using device: {device}")

# --- Функции load_data, clean_text, extract_text, extract_message_data, save_to_csv (save_to_csv использует обновленный RESULTS_DIR) ---
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        exit()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.replace('\n', ' ').replace('\r', ' ').strip()

      
def extract_text(message):
    text_content = message.get("text", "")
    if isinstance(text_content, str):
        return text_content
    print(f"Warning: Unexpected type for 'text' field in message id {message.get('id')}: {type(text_content)}. Expected str.")
    return "" 

    

def extract_message_data(message):
    extracted_text = extract_text(message)
    if not clean_text(extracted_text):
        return None
    return {
        "text": clean_text(extracted_text),
        "date": message.get("date", ""),
        "from": message.get("from", ""),
        "photo": message.get("photo", ""),
        "file_name": message.get("file_name", "")
    }

def save_to_csv(data, filename): # filename - это имя файла, например 'posts_data.csv'
    if not data:
        print("Warning: No data to save to CSV.")
        return
    filepath = os.path.join(RESULTS_DIR, filename) # RESULTS_DIR теперь глобально определен и указывает на model/torch_text/
    try:
        with open(filepath, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Data saved to {filepath}")
    except IOError as e:
        print(f"Error saving data to CSV {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving to CSV: {e}")

# Загрузка и обработка данных
def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)
    if not ad_data or 'messages' not in ad_data:
        print(f"Error: No messages found or invalid format in {ad_filepath}")
        ad_texts_data = []
    else:
        ad_texts_data = [extract_message_data(msg) for msg in ad_data['messages']]
        ad_texts_data = [item for item in ad_texts_data if item is not None]
    if not non_ad_data or 'messages' not in non_ad_data:
        print(f"Error: No messages found or invalid format in {non_ad_filepath}")
        non_ad_texts_data = []
    else:
        non_ad_texts_data = [extract_message_data(msg) for msg in non_ad_data['messages']]
        non_ad_texts_data = [item for item in non_ad_texts_data if item is not None]
    if not ad_texts_data and not non_ad_texts_data:
        print("Error: No valid messages found. Exiting.")
        exit()
    all_data_for_csv = ad_texts_data + non_ad_texts_data
    if all_data_for_csv:
        save_to_csv(all_data_for_csv, 'posts_data.csv') # Сохранит в model/torch_text/posts_data.csv
    else:
        print("Warning: No valid data to save in posts_data.csv")
    texts = [msg['text'] for msg in ad_texts_data] + [msg['text'] for msg in non_ad_texts_data]
    labels = [1] * len(ad_texts_data) + [0] * len(non_ad_texts_data)
    if not texts:
        print("Error: No text data extracted. Exiting.")
        exit()
    print(f"Total texts processed: {len(texts)}, Ads: {len(ad_texts_data)}, Non-Ads: {len(non_ad_texts_data)}")
    return texts, labels

# Класс Dataset для PyTorch
class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        try:
            vector = self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
        except Exception as e:
            print(f"Error vectorizing text at index {idx}: '{text[:50]}...'. Error: {e}")
            vocab_size = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 20000
            vector = np.zeros(vocab_size, dtype=np.float32)

        return {
            'text': torch.tensor(vector, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Модель AdvancedTextClassifier
class AdvancedTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers=[512, 256, 128], num_classes=2,
                 dropout=0.3, activation='relu', use_batch_norm=True):
        super(AdvancedTextClassifier, self).__init__()
        layers = []
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm: layers.append(nn.BatchNorm1d(hidden_size))
            if activation == 'relu': layers.append(nn.ReLU())
            elif activation == 'leaky_relu': layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu': layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        current_batch_size = x.size(0)
        processed_x = x
        for layer in self.hidden_layers:
            if isinstance(layer, nn.BatchNorm1d) and current_batch_size <= 1: continue
            processed_x = layer(processed_x)
        x = self.output_layer(processed_x)
        return x

# Функции train_model, validate_model
def train_model(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    processed_samples = 0
    for batch in dataloader:
        if any(isinstance(layer, nn.BatchNorm1d) for layer in model.hidden_layers if hasattr(model, 'hidden_layers')) and batch['text'].size(0) <= 1: continue
        inputs = batch['text'].to(device)
        labels = batch['label'].to(device)
        batch_size = inputs.size(0)
        processed_samples += batch_size
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    if processed_samples == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
    epoch_loss = running_loss / processed_samples
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    processed_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            batch_size = inputs.size(0)
            processed_samples += batch_size
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if processed_samples == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
    epoch_loss = running_loss / processed_samples
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

# Функция для оптимизации гиперпараметров
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 12 , step=1)
    hidden_sizes = []
    last_hidden_size = trial.suggest_int('hidden_size_0', 64, 2048, step=64)
    hidden_sizes.append(last_hidden_size)
    for i in range(1, num_layers):
         last_hidden_size = trial.suggest_int(f'hidden_size_{i}', 64, last_hidden_size)
         hidden_sizes.append(last_hidden_size)
    dropout = trial.suggest_float('dropout', 0.5, 0.7, step=0.025)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512, 1024, 2048])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad'])
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-1, log=True)

    model = AdvancedTextClassifier(
        input_size=train_vectors.shape[1],
        hidden_layers=hidden_sizes, dropout=dropout, activation=activation, use_batch_norm=use_batch_norm
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'AdamW': optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop': optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else: optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # fallback
    num_epochs_max = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_max)

    train_dataset = TextDataset(train_texts, train_labels, vectorizer)
    val_dataset = TextDataset(val_texts, val_labels, vectorizer)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    drop_last_train = use_batch_norm
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        drop_last=drop_last_train, num_workers=num_workers, pin_memory=use_gpu
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu
    )

    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    temp_model_dir = os.path.join(RESULTS_DIR, "temp_models") # Сохранит в model/torch_text/temp_models/
    os.makedirs(temp_model_dir, exist_ok=True)
    temp_model_path = os.path.join(temp_model_dir, f'temp_best_model_trial_{trial.number}.pth')
    model_saved = False

    for epoch in range(num_epochs_max):
        train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_precision, val_recall = validate_model(model, val_loader, criterion, device)
        if scheduler: scheduler.step()
        trial.report(val_f1, epoch)
        if trial.should_prune():
             if model_saved and os.path.exists(temp_model_path):
                 try: os.remove(temp_model_path)
                 except OSError as e: print(f"W: Could not remove {temp_model_path}: {e}")
             raise optuna.TrialPruned()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            try:
                torch.save(model.state_dict(), temp_model_path)
                model_saved = True
            except Exception as e:
                print(f"E: saving model trial {trial.number}: {e}")
                model_saved = False
        else:
            patience_counter += 1
            if patience_counter >= patience: break

    if model_saved and os.path.exists(temp_model_path):
        try: pass
        except Exception as e: print(f"W: loading {temp_model_path}: {e}.")
        finally:
             try: os.remove(temp_model_path)
             except OSError as e: print(f"W: removing {temp_model_path}: {e}")
    elif os.path.exists(temp_model_path):
         try:
              os.remove(temp_model_path)
              print(f"W: Removed {temp_model_path} (not marked saved).")
         except OSError as e: print(f"W: removing {temp_model_path}: {e}")
    return best_val_f1


# Основной код
if __name__ == "__main__":
    start_time = time.time()

    # --- НАЧАЛО ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---
    # Пути к данным теперь строятся от PROJECT_ROOT
    ad_filepath = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ads_unified.json')
    non_ad_filepath = os.path.join(PROJECT_ROOT, 'data', 'processed', 'non_ads_unified.json')
    # --- КОНЕЦ ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---

    texts, labels = process_data(ad_filepath, non_ad_filepath)

    if len(texts) < 10: print("E: Not enough data to split."); exit()
    if len(set(labels)) < 2: print("W: Only one class present."); stratify_param = None
    else: stratify_param = labels
    try:
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=stratify_param
        )
        if len(train_val_texts) < 2:
            print("W: Not enough data for validation split.")
            train_texts, val_texts = train_val_texts, train_val_texts
            train_labels, val_labels = train_val_labels, train_val_labels
            val_stratify_param = None
        elif len(set(train_val_labels)) < 2:
             print("W: Only one class in train_val set.")
             val_stratify_param = None
             train_texts, val_texts, train_labels, val_labels = train_test_split(
                 train_val_texts, train_val_labels, test_size=0.20, random_state=42, stratify=val_stratify_param
             )
        else:
             val_stratify_param = train_val_labels
             train_texts, val_texts, train_labels, val_labels = train_test_split(
                 train_val_texts, train_val_labels, test_size=0.20, random_state=42, stratify=val_stratify_param
             )
    except ValueError as e: print(f"E: Data splitting: {e}."); exit()
    print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}")

    try:
        try:
             _ = stopwords.words('russian')
        except LookupError:
             print("Downloading nltk stopwords...")
             nltk.download('stopwords', quiet=False)
        russian_stop_words = stopwords.words('russian')
    except Exception as e:
        print(f"Warning: Could not download or load NLTK stopwords. Proceeding without them. Error: {e}")
        russian_stop_words = None

    vectorizer = TfidfVectorizer(
        max_features=20000, stop_words=russian_stop_words, ngram_range=(1, 2),
        min_df=3, max_df=0.9, token_pattern=r'(?u)\b\w\w+\b'
    )
    print("Fitting vectorizer...")
    train_vectors = vectorizer.fit_transform(train_texts)
    val_vectors = vectorizer.transform(val_texts)
    test_vectors = vectorizer.transform(test_texts)
    print(f"Vectorizer fitted. Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train vectors shape: {train_vectors.shape}")

    class_counts = Counter(train_labels)
    if len(class_counts) < 2:
         print("W: Only one class found in training data.")
         class_weights_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    else:
        total_samples = len(train_labels)
        weight_class_0 = total_samples / (2.0 * class_counts.get(0, 1))
        weight_class_1 = total_samples / (2.0 * class_counts.get(1, 1))
        class_weights_tensor = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32).to(device)
    sample_weights = [class_weights_tensor[label].item() for label in train_labels] # .item() to get Python numbers
    print(f"Class counts in train set: {class_counts}")
    print(f"Calculated class weights for sampler: {class_weights_tensor.cpu().numpy()}")

    sampler = TPESampler(n_startup_trials=15, multivariate=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    print("\nStarting hyperparameter optimization...")
    opt_start_time = time.time()
    try:
         study.optimize(objective, n_trials=5, timeout=18000, n_jobs=1) 
    except Exception as e:
         print(f"\nOptimization stopped due to an error: {e}")
         import traceback
         traceback.print_exc()
    opt_duration = time.time() - opt_start_time
    print(f"Optimization finished in {opt_duration:.2f} seconds.")

    if not study.trials or not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
         print("\nNo trials completed successfully. Cannot train final model.")
         exit()
    try:
        best_trial = study.best_trial
        print(f"\nBest trial number: {best_trial.number}")
        print(f"Best value (Validation F1): {best_trial.value:.4f}")
        print("Best parameters found:")
        best_params = best_trial.params
        for key, value in best_params.items(): print(f"  {key}: {value}")
    except ValueError:
        print("\nNo successful trials found to determine best parameters.")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
             print("Using parameters from the last completed trial as fallback.")
             best_params = completed_trials[-1].params
             print("Fallback parameters:")
             for key, value in best_params.items(): print(f"  {key}: {value}")
        else: print("No completed trials found. Exiting."); exit()

    num_layers = best_params['num_layers']
    hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(num_layers)]
    print("\nTraining final model with best parameters...")
    final_model = AdvancedTextClassifier(
        input_size=train_vectors.shape[1], hidden_layers=hidden_sizes, dropout=best_params['dropout'],
        activation=best_params['activation'], use_batch_norm=best_params['use_batch_norm']
    ).to(device)
    criterion = nn.CrossEntropyLoss() # можно добавить .to(device) если есть веса
    optimizer_name = best_params['optimizer']
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    batch_size = best_params['batch_size']
    if optimizer_name == 'AdamW': optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop': optimizer = optim.RMSprop(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else: optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay) # fallback

    all_train_texts = train_texts + val_texts
    all_train_labels = train_labels + val_labels
    print(f"Training final model on {len(all_train_texts)} samples.")

    all_class_counts = Counter(all_train_labels)
    if len(all_class_counts) < 2: all_class_weights_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    else:
        total_samples_all = len(all_train_labels)
        weight_class_0_all = total_samples_all / (2.0 * all_class_counts.get(0, 1))
        weight_class_1_all = total_samples_all / (2.0 * all_class_counts.get(1, 1))
        all_class_weights_tensor = torch.tensor([weight_class_0_all, weight_class_1_all], dtype=torch.float32).to(device)
    all_sample_weights = [all_class_weights_tensor[label].item() for label in all_train_labels] # .item()

    all_train_dataset = TextDataset(all_train_texts, all_train_labels, vectorizer)
    test_dataset = TextDataset(test_texts, test_labels, vectorizer)
    all_train_sampler = WeightedRandomSampler(weights=all_sample_weights, num_samples=len(all_train_dataset), replacement=True)
    drop_last_final = best_params['use_batch_norm']
    final_train_loader = DataLoader(
        all_train_dataset, batch_size=batch_size, sampler=all_train_sampler,
        drop_last=drop_last_final, num_workers=num_workers, pin_memory=use_gpu
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu
    )

    num_epochs_final = 25
    scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_final)
    best_test_f1 = -1.0
    optimal_threshold = 0.5 # Инициализация перед использованием

    # --- НАЧАЛО ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---
    # Пути для сохранения финальной модели, векторизатора и параметров
    final_model_path = os.path.join(RESULTS_DIR, 'best_final_model.pth')
    vectorizer_path = os.path.join(RESULTS_DIR, 'final_vectorizer.pkl')
    best_params_path = os.path.join(RESULTS_DIR, 'final_best_params.json')
    # --- КОНЕЦ ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---

    final_train_start_time = time.time()
    for epoch in range(num_epochs_final):
        epoch_start_time = time.time()
        train_loss, train_acc, train_f1, _, _ = train_model(final_model, final_train_loader, criterion, optimizer, device)
        if scheduler_final: scheduler_final.step()
        test_loss, test_acc, test_f1, test_precision, test_recall = validate_model(final_model, test_loader, criterion, device)
        epoch_duration = time.time() - epoch_start_time
        print(f"\n--- Final Training Epoch {epoch+1}/{num_epochs_final} ({epoch_duration:.2f}s) ---")
        print(f"Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Test  -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Prec: {test_precision:.4f} | Rec: {test_recall:.4f}")
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            try:
                # Определение optimal_threshold должно произойти до сохранения,
                # если мы хотим сохранить его актуальное значение.
                # Пока что сохраняем предварительное значение 0.5, позже оно обновится.
                # Если нужно сохранить актуальный optimal_threshold, его нужно вычислить здесь
                # или сохранить модель после цикла анализа порога.
                # Для простоты, сохраняем с предварительным, или можно вынести сохранение
                # порога в отдельный файл после его определения.
                # Пока что: optimal_threshold = 0.5 (уже инициализировано)
                # ВАЖНО: optimal_threshold здесь будет 0.5. Если нужно актуальное,
                # то сохранение модели/конфига с порогом должно быть ПОСЛЕ его вычисления.
                # Либо сохранить модель БЕЗ порога, а порог сохранить отдельно.
                # Я оставлю как есть, но это место для возможного улучшения.

                # Исправленный save: 'vectorizer' не должен быть здесь, он сохраняется через joblib.
                # 'threshold' здесь будет 0.5, так как оптимальный еще не вычислен.
                # Если важно сохранить лучший ОПТИМАЛЬНЫЙ порог, то эту часть нужно делать
                # ПОСЛЕ вычисления порога. Или сохранить модель без порога, а порог отдельно.
                torch.save({
                    'model_config': {
                        'input_size': train_vectors.shape[1],
                        'hidden_layers': hidden_sizes,
                        'dropout': best_params['dropout'],
                        'activation': best_params['activation'],
                        'use_batch_norm': best_params['use_batch_norm'],
                        'num_classes': 2
                    },
                    'model_state': final_model.state_dict(),
                    # 'vectorizer': vectorizer, # Не сохраняем векторизатор с моделью torch, используем joblib
                    'threshold': optimal_threshold # Это будет 0.5 на данном этапе
                }, final_model_path)
                joblib.dump(vectorizer, vectorizer_path)
                with open(best_params_path, 'w') as f: json.dump(best_params, f, indent=4)
                print(f"Epoch {epoch+1}: New best model saved with Test F1: {best_test_f1:.4f} (Model: {final_model_path})")
            except Exception as e: print(f"E: saving final model/etc: {e}")
    final_train_duration = time.time() - final_train_start_time
    print(f"\nFinal training finished in {final_train_duration:.2f} seconds. Best Test F1 Score: {best_test_f1:.4f}")

    if os.path.exists(final_model_path):
         print("\nLoading best saved final model for evaluation and threshold analysis...")
         try:
             # Загружаем сохраненную модель
             checkpoint = torch.load(final_model_path, map_location=device)
             model_config = checkpoint['model_config']
             # Пересоздаем модель с сохраненной конфигурацией
             final_model_reloaded = AdvancedTextClassifier(
                input_size=model_config['input_size'],
                hidden_layers=model_config['hidden_layers'],
                dropout=model_config['dropout'],
                activation=model_config['activation'],
                use_batch_norm=model_config['use_batch_norm'],
                num_classes=model_config['num_classes']
             ).to(device)
             final_model_reloaded.load_state_dict(checkpoint['model_state'])
             final_model_reloaded.eval() # Важно перевести в режим оценки

             # Векторизатор загружается отдельно
             # vectorizer = joblib.load(vectorizer_path) # Уже загружен и доступен глобально

             all_test_preds, all_test_labels, all_test_positive_probs = [], [], []
             with torch.no_grad():
                 for batch in test_loader:
                     inputs = batch['text'].to(device)
                     labels = batch['label'].to(device)
                     outputs = final_model_reloaded(inputs) # Используем перезагруженную модель
                     probabilities = torch.softmax(outputs, dim=1)
                     positive_probs = probabilities[:, 1]
                     _, predicted = torch.max(outputs.data, 1)
                     all_test_preds.extend(predicted.cpu().numpy())
                     all_test_labels.extend(labels.cpu().numpy())
                     all_test_positive_probs.extend(positive_probs.cpu().numpy())

             all_test_labels = np.array(all_test_labels)
             all_test_preds = np.array(all_test_preds)
             all_test_positive_probs = np.array(all_test_positive_probs)

             print("\nFinal Report (Test Set, threshold=0.5):")
             print(classification_report(all_test_labels, all_test_preds, target_names=['Non-Ad (0)', 'Ad (1)'], zero_division=0))

             if len(np.unique(all_test_labels)) > 1 and len(all_test_positive_probs) > 0:
                 print("\nPerforming threshold analysis...")
                 precision_vals, recall_vals, thresholds_prc = precision_recall_curve(all_test_labels, all_test_positive_probs)
                 # Убираем последний элемент recall и precision, т.к. thresholds короче на 1
                 f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-9)
                 
                 if len(f1_scores) > 0:
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds_prc[optimal_idx] # Используем thresholds_prc
                    optimal_f1 = f1_scores[optimal_idx]
                 else: # Если f1_scores пуст, например, из-за единственного класса
                    optimal_threshold = 0.5
                    optimal_f1 = f1_score(all_test_labels, (all_test_positive_probs >= 0.5).astype(int), zero_division=0)


                 print(f"Optimal threshold: {optimal_threshold:.4f} (F1 = {optimal_f1:.4f})")
                 optimal_preds = (all_test_positive_probs >= optimal_threshold).astype(int)
                 print(f"\nFinal Report (Test Set, optimal threshold={optimal_threshold:.4f}):")
                 print(classification_report(all_test_labels, optimal_preds, target_names=['Non-Ad (0)', 'Ad (1)'], zero_division=0))
                 
                 # Обновляем модель с новым порогом, если это нужно
                 try:
                    torch.save({
                        'model_config': model_config,
                        'model_state': final_model_reloaded.state_dict(),
                        'threshold': optimal_threshold # Сохраняем актуальный порог
                    }, final_model_path)
                    print(f"Model at {final_model_path} updated with optimal threshold: {optimal_threshold:.4f}")
                 except Exception as e:
                    print(f"E: updating model with optimal threshold: {e}")

                 plt.figure(figsize=(10, 7))
                 plt.plot(thresholds_prc, precision_vals[:-1], label='Precision', lw=2)
                 plt.plot(thresholds_prc, recall_vals[:-1], label='Recall', lw=2)
                 plt.plot(thresholds_prc, f1_scores, label='F1-Score', lw=2, linestyle=':') # f1_scores уже правильной длины
                 plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Thr={optimal_threshold:.2f}, F1={optimal_f1:.2f}')
                 plt.xlabel('Threshold'); plt.ylabel('Score'); plt.title('Precision, Recall, F1 vs. Threshold')
                 plt.legend(loc='best'); plt.grid(True); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
                 
                 # --- НАЧАЛО ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---
                 plot_path = os.path.join(RESULTS_DIR, 'threshold_analysis.png') # Сохранит в model/torch_text/
                 # --- КОНЕЦ ИЗМЕНЕНИЙ ДЛЯ НОВОЙ СТРУКТУРЫ ---
                 try:
                      plt.savefig(plot_path); print(f"Threshold plot saved: {plot_path}"); plt.close()
                 except Exception as e: print(f"E: saving threshold plot: {e}")
             else: print("\nSkipping threshold analysis (not enough classes or probabilities).")
         except Exception as e:
             print(f"E: loading/evaluating final model: {e}"); import traceback; traceback.print_exc()
    else: print(f"Final model not found: {final_model_path}.")

    total_duration = time.time() - start_time
    print(f"\nTotal script duration: {total_duration:.2f} seconds.")
    print("\nScript finished.")