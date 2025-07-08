import json
import csv
import os
import time
from datetime import datetime
from collections import Counter
# --- ИЗМЕНЕНИЕ: Добавляем multiprocessing ---
import multiprocessing

import nltk
from nltk.corpus import stopwords
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

# --- Структура проекта (без изменений) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "model", "torch_text")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Настройка оборудования (без изменений) ---
# num_workers теперь будет определяться внутри main, чтобы избежать проблем с multiprocessing
device = torch.device('cpu')
use_gpu = False
print(f"Using device: {device}")


# --- Функции загрузки и обработки данных (без изменений) ---
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
    text_content = message.get("text_content", "")
    if isinstance(text_content, str):
        return text_content
    print(f"Warning: Unexpected type for 'text_content' in message id {message.get('id')}: {type(text_content)}. Expected str.")
    return ""

def extract_message_data(message):
    extracted_text = extract_text(message)
    if not clean_text(extracted_text):
        return None
    photo_path = ""
    for att in message.get("attachments", []):
        if att.get("type") == "photo":
            photo_path = att.get("path", "")
            break
    return {
        "text": clean_text(extracted_text),
        "date": message.get("date_unixtime", ""),
        "from": message.get("from_id", ""),
        "photo": photo_path,
        "file_name": photo_path
    }

def save_to_csv(data, filename):
    if not data:
        print("Warning: No data to save to CSV.")
        return
    filepath = os.path.join(RESULTS_DIR, filename)
    try:
        with open(filepath, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Data saved to {filepath}")
    except IOError as e: print(f"Error saving data to CSV {filepath}: {e}")
    except Exception as e: print(f"An unexpected error occurred while saving to CSV: {e}")

def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)
    if not ad_data or 'messages' not in ad_data: ad_texts_data = []
    else: ad_texts_data = [d for d in [extract_message_data(m) for m in ad_data['messages']] if d]
    if not non_ad_data or 'messages' not in non_ad_data: non_ad_texts_data = []
    else: non_ad_texts_data = [d for d in [extract_message_data(m) for m in non_ad_data['messages']] if d]
    if not ad_texts_data and not non_ad_texts_data: print("Error: No valid messages found."); exit()
    all_data_for_csv = ad_texts_data + non_ad_texts_data
    if all_data_for_csv: save_to_csv(all_data_for_csv, 'posts_data.csv')
    texts = [msg['text'] for msg in ad_texts_data] + [msg['text'] for msg in non_ad_texts_data]
    labels = [1] * len(ad_texts_data) + [0] * len(non_ad_texts_data)
    if not texts: print("Error: No text data extracted."); exit()
    print(f"Total texts processed: {len(texts)}, Ads: {len(ad_texts_data)}, Non-Ads: {len(non_ad_texts_data)}")
    return texts, labels

# Класс Dataset и Модель (без изменений)
class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.texts = texts; self.labels = labels; self.vectorizer = vectorizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]; label = self.labels[idx]
        try: vector = self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
        except Exception as e:
            vocab_size = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 20000
            vector = np.zeros(vocab_size, dtype=np.float32)
        return {'text': torch.tensor(vector, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}

class AdvancedTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers=[512, 256, 128], num_classes=2, dropout=0.3, activation='relu', use_batch_norm=True):
        super(AdvancedTextClassifier, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm: layers.append(nn.BatchNorm1d(hidden_size))
            if activation == 'relu': layers.append(nn.ReLU())
            elif activation == 'leaky_relu': layers.append(nn.LeakyReLU(0.1))
            else: layers.append(nn.ELU())
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
        return self.output_layer(processed_x)

# Функции train_model, validate_model (без изменений)
def train_model(model, dataloader, criterion, optimizer, device):
    model.train(); running_loss = 0.0; all_preds, all_labels = [], []; processed_samples = 0
    for batch in dataloader:
        if any(isinstance(layer, nn.BatchNorm1d) for layer in model.hidden_layers if hasattr(model, 'hidden_layers')) and batch['text'].size(0) <= 1: continue
        inputs = batch['text'].to(device); labels = batch['label'].to(device); batch_size = inputs.size(0)
        processed_samples += batch_size
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    if processed_samples == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
    return (running_loss / processed_samples, accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, zero_division=0), precision_score(all_labels, all_preds, zero_division=0),
            recall_score(all_labels, all_preds, zero_division=0))

def validate_model(model, dataloader, criterion, device):
    model.eval(); running_loss = 0.0; all_preds, all_labels = [], []; processed_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['text'].to(device); labels = batch['label'].to(device); batch_size = inputs.size(0)
            processed_samples += batch_size
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    if processed_samples == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
    return (running_loss / processed_samples, accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, zero_division=0), precision_score(all_labels, all_preds, zero_division=0),
            recall_score(all_labels, all_preds, zero_division=0))

# --- ИЗМЕНЕНИЕ: Функция objective теперь принимает dataloader'ы как аргументы ---
def objective(trial, train_dataset, val_dataset, sample_weights, num_workers):
    # Гиперпараметры (без изменений)
    num_layers = trial.suggest_int('num_layers', 1, 12, step=1)
    hidden_sizes = []
    last_hidden_size = trial.suggest_int('hidden_size_0', 64, 2048, step=64)
    hidden_sizes.append(last_hidden_size)
    for i in range(1, num_layers): last_hidden_size = trial.suggest_int(f'hidden_size_{i}', 64, last_hidden_size); hidden_sizes.append(last_hidden_size)
    dropout = trial.suggest_float('dropout', 0.5, 0.7, step=0.025)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512, 1024, 2048])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad'])
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-1, log=True)

    model = AdvancedTextClassifier(input_size=train_dataset.vectorizer.max_features, hidden_layers=hidden_sizes, dropout=dropout, activation=activation, use_batch_norm=use_batch_norm).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'AdamW': optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop': optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else: optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # --- ИЗМЕНЕНИЕ: Создаем dataloader'ы ОДИН РАЗ ЗДЕСЬ ---
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=use_batch_norm, num_workers=num_workers, pin_memory=use_gpu, persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu, persistent_workers=True if num_workers > 0 else False)

    num_epochs_max = 40; scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_max)
    best_val_f1 = 0.0; patience = 5; patience_counter = 0

    for epoch in range(num_epochs_max):
        # Передаем уже созданные loader'ы
        train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_precision, val_recall = validate_model(model, val_loader, criterion, device)
        if scheduler: scheduler.step()
        trial.report(val_f1, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break
            
    return best_val_f1


# Основной код
if __name__ == "__main__":
    # --- ИЗМЕНЕНИЕ: Установка метода запуска multiprocessing для кросс-платформенности ---
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Может быть вызвано, если метод уже установлен. Игнорируем.
        pass

    start_time = time.time()

    # Определяем num_workers здесь
    try: num_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 1
    except NotImplementedError: num_workers = 4
    print(f"Using num_workers = {num_workers} for DataLoaders.")
    
    ad_filepath = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ads_unified.json')
    non_ad_filepath = os.path.join(PROJECT_ROOT, 'data', 'processed', 'non_ads_unified.json')
    texts, labels = process_data(ad_filepath, non_ad_filepath)

    if len(texts) < 10: print("E: Not enough data to split."); exit()
    stratify_param = labels if len(set(labels)) > 1 else None
    
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=stratify_param)
    val_stratify_param = train_val_labels if len(set(train_val_labels)) > 1 else None
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=0.20, random_state=42, stratify=val_stratify_param)
    print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}")

    try:
        russian_stop_words = stopwords.words('russian')
    except LookupError:
        print("Downloading nltk stopwords...")
        nltk.download('stopwords')
        russian_stop_words = stopwords.words('russian')

    vectorizer = TfidfVectorizer(max_features=20000, stop_words=russian_stop_words, ngram_range=(1, 2), min_df=3, max_df=0.9, token_pattern=r'(?u)\b\w\w+\b')
    print("Fitting vectorizer...")
    train_vectors = vectorizer.fit_transform(train_texts)
    print(f"Vectorizer fitted. Vocabulary size: {len(vectorizer.vocabulary_)}")

    # --- ИЗМЕНЕНИЕ: Создаем датасеты один раз перед оптимизацией ---
    train_dataset_opt = TextDataset(train_texts, train_labels, vectorizer)
    val_dataset_opt = TextDataset(val_texts, val_labels, vectorizer)

    class_counts = Counter(train_labels)
    if len(class_counts) < 2: class_weights_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    else:
        weight_class_0 = len(train_labels) / (2.0 * class_counts.get(0, 1)); weight_class_1 = len(train_labels) / (2.0 * class_counts.get(1, 1))
        class_weights_tensor = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32).to(device)
    sample_weights = [class_weights_tensor[label].item() for label in train_labels]
    print(f"Class counts in train set: {class_counts}"); print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")

    sampler = TPESampler(n_startup_trials=15, multivariate=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    print("\nStarting hyperparameter optimization...")
    opt_start_time = time.time()
    try:
        # --- ИЗМЕНЕНИЕ: Передаем датасеты и веса в objective ---
        study.optimize(lambda trial: objective(trial, train_dataset_opt, val_dataset_opt, sample_weights, num_workers), n_trials=2, timeout=18000, n_jobs=1)
    except Exception as e: print(f"\nOptimization stopped due to an error: {e}"); import traceback; traceback.print_exc()
    opt_duration = time.time() - opt_start_time
    print(f"Optimization finished in {opt_duration:.2f} seconds.")

    # ... (Остальная часть кода для обучения финальной модели и оценки остается такой же)
    # ... (Но создание DataLoader'ов там тоже вынесено за цикл, что уже было правильно)
    # ... Я приведу этот блок тоже, чтобы код был полным.
    if not study.trials or not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
         print("\nNo trials completed successfully. Cannot train final model."); exit()
    try:
        best_trial = study.best_trial
        print(f"\nBest trial number: {best_trial.number}"); print(f"Best value (Validation F1): {best_trial.value:.4f}")
        print("Best parameters found:"); best_params = best_trial.params
        for key, value in best_params.items(): print(f"  {key}: {value}")
    except ValueError:
        print("\nNo successful trials found."); exit()

    num_layers = best_params['num_layers']; hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(num_layers)]
    print("\nTraining final model with best parameters...")
    final_model = AdvancedTextClassifier(input_size=vectorizer.max_features, hidden_layers=hidden_sizes, dropout=best_params['dropout'], activation=best_params['activation'], use_batch_norm=best_params['use_batch_norm']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_name = best_params['optimizer']; learning_rate = best_params['learning_rate']; weight_decay = best_params['weight_decay']; batch_size = best_params['batch_size']
    if optimizer_name == 'AdamW': optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop': optimizer = optim.RMSprop(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else: optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    all_train_texts = train_texts + val_texts; all_train_labels = train_labels + val_labels
    print(f"Training final model on {len(all_train_texts)} samples.")

    all_class_counts = Counter(all_train_labels)
    if len(all_class_counts) < 2: all_class_weights_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    else:
        w0 = len(all_train_labels)/(2.0*all_class_counts.get(0,1)); w1 = len(all_train_labels)/(2.0*all_class_counts.get(1,1))
        all_class_weights_tensor = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    all_sample_weights = [all_class_weights_tensor[label].item() for label in all_train_labels]

    all_train_dataset = TextDataset(all_train_texts, all_train_labels, vectorizer)
    test_dataset = TextDataset(test_texts, test_labels, vectorizer)
    all_train_sampler = WeightedRandomSampler(weights=all_sample_weights, num_samples=len(all_train_dataset), replacement=True)
    
    # --- ИЗМЕНЕНИЕ: Создаем DataLoader'ы ОДИН РАЗ перед финальным обучением ---
    final_train_loader = DataLoader(all_train_dataset, batch_size=batch_size, sampler=all_train_sampler, drop_last=best_params['use_batch_norm'], num_workers=num_workers, pin_memory=use_gpu, persistent_workers=True if num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu, persistent_workers=True if num_workers > 0 else False)

    num_epochs_final = 25; scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_final)
    best_test_f1 = -1.0; optimal_threshold = 0.5
    final_model_path = os.path.join(RESULTS_DIR, 'best_final_model.pth')
    vectorizer_path = os.path.join(RESULTS_DIR, 'final_vectorizer.pkl')
    best_params_path = os.path.join(RESULTS_DIR, 'final_best_params.json')

    final_train_start_time = time.time()
    for epoch in range(num_epochs_final):
        epoch_start_time = time.time()
        # Передаем уже созданные loader'ы
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
                torch.save({'model_config': {'input_size': vectorizer.max_features, 'hidden_layers': hidden_sizes, 'dropout': best_params['dropout'], 'activation': best_params['activation'], 'use_batch_norm': best_params['use_batch_norm'], 'num_classes': 2}, 'model_state': final_model.state_dict(), 'threshold': optimal_threshold}, final_model_path)
                joblib.dump(vectorizer, vectorizer_path)
                with open(best_params_path, 'w') as f: json.dump(best_params, f, indent=4)
                print(f"Epoch {epoch+1}: New best model saved with Test F1: {best_test_f1:.4f}")
            except Exception as e: print(f"E: saving final model: {e}")
    final_train_duration = time.time() - final_train_start_time
    print(f"\nFinal training finished in {final_train_duration:.2f} seconds. Best Test F1 Score: {best_test_f1:.4f}")

    if os.path.exists(final_model_path):
         print("\nLoading best saved final model for evaluation...")
         try:
             checkpoint = torch.load(final_model_path, map_location=device)
             model_config = checkpoint['model_config']
             final_model_reloaded = AdvancedTextClassifier(input_size=model_config['input_size'], hidden_layers=model_config['hidden_layers'], dropout=model_config['dropout'], activation=model_config['activation'], use_batch_norm=model_config['use_batch_norm'], num_classes=model_config['num_classes']).to(device)
             final_model_reloaded.load_state_dict(checkpoint['model_state'])
             final_model_reloaded.eval()

             all_test_preds, all_test_labels, all_test_positive_probs = [], [], []
             with torch.no_grad():
                 for batch in test_loader:
                     inputs = batch['text'].to(device); labels = batch['label'].to(device)
                     outputs = final_model_reloaded(inputs)
                     probabilities = torch.softmax(outputs, dim=1)
                     all_test_preds.extend(torch.max(outputs.data, 1)[1].cpu().numpy())
                     all_test_labels.extend(labels.cpu().numpy())
                     all_test_positive_probs.extend(probabilities[:, 1].cpu().numpy())
             
             print("\nFinal Report (Test Set, threshold=0.5):"); print(classification_report(all_test_labels, np.array(all_test_preds), target_names=['Non-Ad (0)', 'Ad (1)'], zero_division=0))
             
             if len(np.unique(all_test_labels)) > 1:
                 print("\nPerforming threshold analysis...")
                 precision_vals, recall_vals, thresholds_prc = precision_recall_curve(all_test_labels, all_test_positive_probs)
                 f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-9)
                 if len(f1_scores) > 0:
                    optimal_idx = np.argmax(f1_scores); optimal_threshold = thresholds_prc[optimal_idx]; optimal_f1 = f1_scores[optimal_idx]
                 else: optimal_threshold = 0.5; optimal_f1 = f1_score(all_test_labels, (np.array(all_test_positive_probs) >= 0.5).astype(int))
                 print(f"Optimal threshold: {optimal_threshold:.4f} (F1 = {optimal_f1:.4f})")
                 optimal_preds = (np.array(all_test_positive_probs) >= optimal_threshold).astype(int)
                 print(f"\nFinal Report (Test Set, optimal threshold={optimal_threshold:.4f}):"); print(classification_report(all_test_labels, optimal_preds, target_names=['Non-Ad (0)', 'Ad (1)'], zero_division=0))
                 try:
                    checkpoint['threshold'] = optimal_threshold
                    torch.save(checkpoint, final_model_path)
                    print(f"Model at {final_model_path} updated with optimal threshold: {optimal_threshold:.4f}")
                 except Exception as e: print(f"E: updating model with optimal threshold: {e}")
                 
                 plt.figure(figsize=(10, 7)); plt.plot(thresholds_prc, precision_vals[:-1], label='Precision', lw=2); plt.plot(thresholds_prc, recall_vals[:-1], label='Recall', lw=2); plt.plot(thresholds_prc, f1_scores, label='F1-Score', lw=2, linestyle=':')
                 plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Thr={optimal_threshold:.2f}, F1={optimal_f1:.2f}'); plt.xlabel('Threshold'); plt.ylabel('Score'); plt.title('Precision, Recall, F1 vs. Threshold')
                 plt.legend(loc='best'); plt.grid(True); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0]);
                 plot_path = os.path.join(RESULTS_DIR, 'threshold_analysis.png')
                 try: plt.savefig(plot_path); print(f"Threshold plot saved: {plot_path}"); plt.close()
                 except Exception as e: print(f"E: saving threshold plot: {e}")
             else: print("\nSkipping threshold analysis.")
         except Exception as e: print(f"E: loading/evaluating final model: {e}"); import traceback; traceback.print_exc()
    
    total_duration = time.time() - start_time
    print(f"\nTotal script duration: {total_duration:.2f} seconds."); print("\nScript finished.")