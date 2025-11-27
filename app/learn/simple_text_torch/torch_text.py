import json
import os
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
from sklearn.metrics import f1_score
import optuna
from collections import Counter
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
    return texts, labels

# --- Dataset (Оптимизированный) ---
class PrecomputedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'text': self.features[idx], 'label': self.labels[idx]}

# --- МОДЕЛЬ ---
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
        inputs = batch['text'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['text'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, zero_division=0)

# --- Optuna Objective ---
def objective(trial, train_features, train_labels, val_features, val_labels, input_size, device):
    # Архитектура
    num_layers = trial.suggest_int('num_layers', 1, 4)
    hidden_sizes = []
    for i in range(num_layers):
        hidden_sizes.append(trial.suggest_int(f'hidden_{i}', 64, 1024, step=64))
        
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    # Оптимизация
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    
    # Dataset Setup
    train_ds = PrecomputedDataset(train_features, train_labels)
    val_ds = PrecomputedDataset(val_features, val_labels)

    counts = Counter(train_labels.tolist())
    weights = [1.0/counts[l] for l in train_labels.tolist()]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = AdvancedTextClassifier(
        input_size=input_size,
        hidden_layers=hidden_sizes,
        dropout=dropout,
        activation=activation,
        use_batch_norm=use_batch_norm
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # ИСПРАВЛЕНИЕ ЗДЕСЬ: Убран verbose=False
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for epoch in range(8): 
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_f1 = validate(model, val_loader, device)
        
        # Шаг планировщика
        scheduler.step(val_f1)
        
        trial.report(val_f1, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        
    return val_f1

def prepare_tensor_data(texts, labels, vectorizer, fit=False):
    if fit:
        print("Fitting vectorizer...")
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = vectorizer.transform(texts)
    return torch.tensor(matrix.toarray(), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

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
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.15, random_state=42, stratify=labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels)
    
    try: nltk.download('stopwords')
    except: pass
    stop_words = stopwords.words('russian')
    
    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(
        max_features=20000, 
        stop_words=stop_words, 
        ngram_range=(1, 2), 
        min_df=3, 
        sublinear_tf=True
    )
    
    X_train, y_train = prepare_tensor_data(train_texts, train_labels, vectorizer, fit=True)
    X_val, y_val = prepare_tensor_data(val_texts, val_labels, vectorizer, fit=False)
    X_test, y_test = prepare_tensor_data(test_texts, test_labels, vectorizer, fit=False)
    
    input_size = X_train.shape[1]
    print(f"Input size: {input_size}")

    print("Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, input_size, device), n_trials=10000)
    
    bp = study.best_params
    print(f"Best params: {bp}")
    
    # --- Финальное обучение ---
    print("Training final model...")
    hidden_layers = [bp[f'hidden_{i}'] for i in range(bp['num_layers'])]
    
    final_model = AdvancedTextClassifier(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout=bp['dropout'],
        activation=bp['activation'],
        use_batch_norm=bp['use_batch_norm']
    ).to(device)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=bp['lr'], weight_decay=bp['weight_decay'])
    
    # И ТУТ ТОЖЕ ПРОВЕРЯЕМ (убрал verbose)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    X_full = torch.cat((X_train, X_val), dim=0)
    y_full = torch.cat((y_train, y_val), dim=0)
    full_ds = PrecomputedDataset(X_full, y_full)
    test_ds = PrecomputedDataset(X_test, y_test)
    
    counts = Counter(y_full.tolist())
    weights = [1.0/(counts[l]**0.5) for l in y_full.tolist()]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_loader = DataLoader(full_ds, batch_size=bp['batch_size'], sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bp['batch_size'], shuffle=False, num_workers=0)
    
    best_test_f1 = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(30):
        loss = train_epoch(final_model, train_loader, criterion, optimizer, device)
        test_f1 = validate(final_model, test_loader, device)
        scheduler.step(test_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Test F1: {test_f1:.4f} | LR: {current_lr:.2e}")
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            patience_counter = 0
            
            checkpoint = {
                'model_config': {
                    'input_size': input_size,
                    'hidden_layers': hidden_layers,
                    'dropout': bp['dropout'],
                    'activation': bp['activation'],
                    'use_batch_norm': bp['use_batch_norm'],
                    'num_classes': 2
                },
                'model_state': final_model.state_dict(),
                'threshold': 0.5
            }
            torch.save(checkpoint, os.path.join(RESULTS_DIR, 'best_final_model.pth'))
            joblib.dump(vectorizer, os.path.join(RESULTS_DIR, 'final_vectorizer.pkl'))
            print("  -> Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
            
    print(f"Final Best F1: {best_test_f1}")