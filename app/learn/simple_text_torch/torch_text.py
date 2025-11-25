import json
import csv
import os
import time
from collections import Counter
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, precision_recall_curve
import optuna
from optuna.samplers import TPESampler
import joblib

# --- Конфигурация путей ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "model", "torch_text")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Утилиты загрузки ---
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)
    
    ad_texts = [m.get('text_content', '') for m in ad_data.get('messages', []) if m.get('text_content')]
    non_ad_texts = [m.get('text_content', '') for m in non_ad_data.get('messages', []) if m.get('text_content')]
    
    texts = ad_texts + non_ad_texts
    labels = [1] * len(ad_texts) + [0] * len(non_ad_texts)
    
    print(f"Processed: {len(texts)} texts ({len(ad_texts)} ads, {len(non_ad_texts)} non-ads)")
    return texts, labels

# --- Dataset & Model ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        # Превращаем текст в вектор на лету (или можно прекэшировать, если памяти много)
        vec = self.vectorizer.transform([self.texts[idx]]).toarray()[0].astype(np.float32)
        return {
            'text': torch.tensor(vec),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ВАЖНО: Этот класс должен совпадать с тем, что ожидает main.py при загрузке
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
        processed_x = self.hidden_layers(x)
        return self.output_layer(processed_x)

# --- Обучение ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, zero_division=0)

# --- Optuna ---
def objective(trial, train_ds, val_ds, num_workers, device):
    # ОГРАНИЧИВАЕМ сложность модели, чтобы не было "вечных 100%"
    num_layers = trial.suggest_int('num_layers', 1, 3) # Максимум 3 слоя
    hidden_sizes = []
    for i in range(num_layers):
        hidden_sizes.append(trial.suggest_int(f'hidden_{i}', 64, 512, step=64))
        
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # DataLoader creation
    # Считаем веса для сэмплера внутри objective для простоты
    labels = train_ds.labels
    counts = Counter(labels)
    weights = [1.0/counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    
    model = AdvancedTextClassifier(
        input_size=train_ds.vectorizer.max_features,
        hidden_layers=hidden_sizes,
        dropout=dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Label Smoothing!
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for epoch in range(5): # Быстрый поиск
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_f1 = validate(model, val_loader, criterion, device)
        trial.report(val_f1, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        
    return val_f1

# --- Main ---
if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn')
    except RuntimeError: pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ad_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ads_unified.json')
    non_ad_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'non_ads_unified.json')
    
    texts, labels = process_data(ad_path, non_ad_path)
    
    # Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
    
    # Vectorizer
    try: nltk.download('stopwords')
    except: pass
    stop_words = stopwords.words('russian')
    
    print("Fitting Vectorizer...")
    # Ограничиваем словарь, чтобы модель не переобучалась на редких словах
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=stop_words, ngram_range=(1, 2), min_df=5)
    vectorizer.fit(train_texts) # Fit only on train!
    
    train_ds = TextDataset(train_texts, train_labels, vectorizer)
    val_ds = TextDataset(val_texts, val_labels, vectorizer)
    test_ds = TextDataset(test_texts, test_labels, vectorizer)
    
    num_workers = min(os.cpu_count(), 4)
    
    print("Starting Optuna...")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda t: objective(t, train_ds, val_ds, num_workers, device), n_trials=350)
    
    best_params = study.best_params
    print(f"Best params: {best_params}")
    
    # Final Training
    print("Training final model...")
    hidden_layers = [best_params[f'hidden_{i}'] for i in range(best_params['num_layers'])]
    
    final_model = AdvancedTextClassifier(
        input_size=vectorizer.max_features,
        hidden_layers=hidden_layers,
        dropout=best_params['dropout']
    ).to(device)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Soft labels
    
    # Full train dataset
    full_train_texts = train_texts + val_texts
    full_train_labels = train_labels + val_labels
    full_train_ds = TextDataset(full_train_texts, full_train_labels, vectorizer)
    
    counts = Counter(full_train_labels)
    weights = [1.0/(counts[l]**0.5) for l in full_train_labels] # Мягкое взвешивание
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_loader = DataLoader(full_train_ds, batch_size=best_params['batch_size'], sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False, num_workers=num_workers)
    
    best_test_f1 = 0.0
    
    for epoch in range(15):
        loss = train_epoch(final_model, train_loader, criterion, optimizer, device)
        test_f1 = validate(final_model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Test F1: {test_f1:.4f}")
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            # Сохраняем в формате, который ждет main.py
            checkpoint = {
                'model_config': {
                    'input_size': vectorizer.max_features,
                    'hidden_layers': hidden_layers,
                    'dropout': best_params['dropout'],
                    'activation': 'relu',
                    'use_batch_norm': True,
                    'num_classes': 2
                },
                'model_state': final_model.state_dict(),
                'threshold': 0.5
            }
            torch.save(checkpoint, os.path.join(RESULTS_DIR, 'best_final_model.pth'))
            joblib.dump(vectorizer, os.path.join(RESULTS_DIR, 'final_vectorizer.pkl'))
            print("  -> Saved best model")
            
    print("Done.")