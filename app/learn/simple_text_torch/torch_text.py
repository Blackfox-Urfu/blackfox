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
from tqdm import tqdm

# --- НАСТРОЙКИ ---
# Если True, база данных Optuna будет удаляться при каждом запуске.
# Если False, скрипт продолжит обучение с того места, где остановился в прошлый раз.
RESET_DB = True 

# --- Конфигурация путей ---

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

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

# --- Dataset ---

class PrecomputedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'text': self.features[idx], 'label': self.labels[idx]}

# --- МОДЕЛЬ ---

class AdvancedTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers=[512, 256], num_classes=2, dropout=0.3, activation='relu', use_batch_norm=True):
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
        
        # Gradient Clipping (Предохранитель от взрыва градиента)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_metrics(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total_confidence = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['text'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probabilities = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            
            total_confidence += confidence.sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_conf = total_confidence / len(all_labels)

    return avg_loss, f1, avg_conf

# --- Optuna Objective ---

def objective(trial, train_features, train_labels, val_features, val_labels, input_size, device):
    # Оптимизированные диапазоны (меньше слоев, меньше LR)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_sizes = []
    for i in range(num_layers):
        hidden_sizes.append(trial.suggest_int(f'hidden_{i}', 128, 512, step=64))

    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(12): 
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_conf = validate_metrics(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        
    return val_loss

def prepare_tensor_data(texts, labels, vectorizer, fit=False):
    if fit:
        print("  Fitting vectorizer (this might take a moment)...")
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
    
    print("\n=== Phase 1/4: Data Loading & Split ===")
    ad_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ads_unified.json')
    non_ad_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'non_ads_unified.json')
    texts, labels = process_data(ad_path, non_ad_path)

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.15, random_state=42, stratify=labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels)

    try: nltk.download('stopwords', quiet=True)
    except: pass
    stop_words = stopwords.words('russian')

    print("\n=== Phase 2/4: Vectorization (Heavy Mode) ===")
    vectorizer = TfidfVectorizer(
        max_features=100000, 
        stop_words=stop_words, 
        ngram_range=(1, 3), 
        min_df=2, 
        sublinear_tf=True
    )

    X_train, y_train = prepare_tensor_data(train_texts, train_labels, vectorizer, fit=True)
    X_val, y_val = prepare_tensor_data(val_texts, val_labels, vectorizer, fit=False)
    X_test, y_test = prepare_tensor_data(test_texts, test_labels, vectorizer, fit=False)

    input_size = X_train.shape[1]
    print(f"  New Input size: {input_size}")

    print("\n=== Phase 3/4: Hyperparameter Optimization ===")
    
    # === НАСТРОЙКА БАЗЫ ДАННЫХ И ЕЕ ОЧИСТКА ===
    db_filename = "optuna_study.db"
    db_path = os.path.join(RESULTS_DIR, db_filename)
    
    # АВТОМАТИЧЕСКАЯ ОЧИСТКА
    if RESET_DB and os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"  [!] Файл {db_filename} удален. Начинаем новую оптимизацию.")
        except PermissionError:
            print(f"  [Error] Не могу удалить {db_filename}. Возможно, он открыт в optuna-dashboard?")
            print("  Пожалуйста, остановите dashboard (Ctrl+C в другом окне) и перезапустите скрипт.")
            exit()

    storage_url = f"sqlite:///{db_path}"
    study_name = "text_classifier_optimization"
    
    print(f"  --> DB stored at: {db_path}")
    print(f"  --> To view dashboard run: optuna-dashboard {storage_url}")
    print("  ---------------------------------------------------------")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction='minimize', 
        pruner=optuna.pruners.HyperbandPruner()
    )
    
    n_trials = 200
    
    # Если мы очистили базу, completed_trials всегда будет 0
    completed_trials = len(study.trials) 
    remaining_trials = max(0, n_trials - completed_trials)

    if remaining_trials > 0:
        with tqdm(total=n_trials, initial=completed_trials, desc="Finding Best Hyperparams", unit="trial") as pbar_opt:
            def optuna_callback(study, trial):
                pbar_opt.update(1)
                best_val = study.best_value if study.best_value else 0.0
                pbar_opt.set_postfix({'Best Loss': f"{best_val:.4f}"})
                
            study.optimize(
                lambda t: objective(t, X_train, y_train, X_val, y_val, input_size, device), 
                n_trials=remaining_trials, 
                callbacks=[optuna_callback]
            )

    bp = study.best_params
    print(f"\nBest params: {bp}")
    print(f"Best Loss: {study.best_value}")

    print("\n=== Phase 4/4: Final Model Training ===")
    hidden_layers = [bp[f'hidden_{i}'] for i in range(bp['num_layers'])]

    final_model = AdvancedTextClassifier(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout=bp['dropout'],
        activation=bp['activation'],
        use_batch_norm=bp['use_batch_norm']
    ).to(device)

    optimizer = optim.AdamW(final_model.parameters(), lr=bp['lr'], weight_decay=bp['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    X_full = torch.cat((X_train, X_val), dim=0)
    y_full = torch.cat((y_train, y_val), dim=0)
    full_ds = PrecomputedDataset(X_full, y_full)
    test_ds = PrecomputedDataset(X_test, y_test)

    counts = Counter(y_full.tolist())
    weights = [1.0/(counts[l]**0.5) for l in y_full.tolist()]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(full_ds, batch_size=bp['batch_size'], sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bp['batch_size'], shuffle=False, num_workers=0)

    best_loss = float('inf')
    best_f1_at_best_loss = 0.0
    patience = 8 
    patience_counter = 0
    max_epochs = 50
    
    with tqdm(range(max_epochs), desc="Training Epochs", unit="epoch") as pbar_train:
        for epoch in pbar_train:
            train_loss = train_epoch(final_model, train_loader, criterion, optimizer, device)
            val_loss, val_f1, val_conf = validate_metrics(final_model, test_loader, criterion, device)
            scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar_train.set_postfix({
                'Loss': f"{val_loss:.4f}", 
                'F1': f"{val_f1:.4f}", 
                'LR': f"{current_lr:.1e}"
            })
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_f1_at_best_loss = val_f1
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
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
    print(f"\nFinal Best Loss: {best_loss:.4f}, F1 at that loss: {best_f1_at_best_loss:.4f}")