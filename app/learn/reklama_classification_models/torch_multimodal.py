import json
import os
import re
import sys
import time
from collections import Counter
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import joblib
import optuna
from tqdm import tqdm
import resource  # Для управления системными лимитами

# --- 1. ПОВЫШЕНИЕ ЛИМИТА ОТКРЫТЫХ ФАЙЛОВ ---
def increase_file_limit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard # Пытаемся взять жесткий лимит системы
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        print(f"✅ System file limit increased from {soft} to {target}")
    except Exception as e:
        print(f"⚠️ Could not increase file limit: {e}")

increase_file_limit()

# Попытка сменить стратегию multiprocessing для экономии дескрипторов
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

# --- ИСПРАВЛЕНИЕ ИМПОРТА ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from .torch_models import MetaLearner
except ImportError:
    try:
        from torch_models import MetaLearner
    except ImportError:
        print("Ошибка: не найден файл torch_models.py с классом MetaLearner")
        sys.exit(1)

# ---------------------------
# --- Конфигурация путей ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
AD_FILE = os.path.join(DATA_DIR, "ads_unified.json")
NON_AD_FILE = os.path.join(DATA_DIR, "non_ads_unified.json")
RAW_DATA_BASE_DIR_AD = os.path.join(PROJECT_ROOT, "data", "raw", "reklama")
RAW_DATA_BASE_DIR_NON_AD = os.path.join(PROJECT_ROOT, "data", "raw", "nereklama")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "model", "multimodal")
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "multimodal_training_cache.pkl")
DB_NAME = "optuna_study.db"
DB_PATH = os.path.join(RESULTS_DIR, DB_NAME)
STORAGE_URL = f"sqlite:///{DB_PATH}"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Аугментация ---
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),               
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Логика извлечения признаков ---
def extract_features_like_server(text: str, has_image: bool) -> list:
    if not isinstance(text, str):
        text = ""
    text_length = len(text)
    link_count = len(re.findall(r'http[s]?://\S+', text))
    mention_count = text.count('@')
    hashtag_count = text.count('#')
    attachment_count = 1 if has_image else 0
    return [text_length, link_count, mention_count, hashtag_count, attachment_count]

# --- Dataset ---
class MultiModalDataset(Dataset):
    def __init__(self, messages, X_text, X_features, image_transform, base_dir_ad, base_dir_non_ad):
        self.messages = messages
        self.X_text = X_text
        self.X_features = X_features
        self.image_transform = image_transform
        self.base_dir_ad = base_dir_ad
        self.base_dir_non_ad = base_dir_non_ad

    def __len__(self): 
        return len(self.messages)

    def __getitem__(self, idx):
        msg = self.messages[idx]
        label = msg['label']
        text_vector = self.X_text[idx].astype(np.float32)
        features_vector = self.X_features[idx].astype(np.float32)
        
        image_tensor = None
        for att in msg.get('attachments', []):
            if att.get('type') == 'photo' and att.get('is_valid'):
                try:
                    base_dir = self.base_dir_ad if label == 1 else self.base_dir_non_ad
                    img_path = os.path.join(base_dir, att['path'])
                    
                    if not os.path.exists(img_path): continue
                    
                    with Image.open(img_path) as img:
                        image = img.convert('RGB')
                        if self.image_transform:
                            image_tensor = self.image_transform(image)
                    break 
                except Exception: 
                    continue
        
        return {
            'text': text_vector, 
            'features': features_vector, 
            'image': image_tensor, 
            'label': label
        }

def collate_fn(batch):
    texts = torch.tensor(np.array([item['text'] for item in batch]), dtype=torch.float32)
    features = torch.tensor(np.array([item['features'] for item in batch]), dtype=torch.float32)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    images = []
    image_indices = []

    for i, item in enumerate(batch):
        if item['image'] is not None:
            images.append(item['image'])
            image_indices.append(i)
            
    collated_batch = {'text': texts, 'features': features, 'labels': labels}
    if images:
        collated_batch['images'] = torch.stack(images)
        collated_batch['image_indices'] = torch.tensor(image_indices, dtype=torch.long)
    return collated_batch

# --- Обучение ---
def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch['text'] = batch['text'].to(device, non_blocking=True)
        batch['features'] = batch['features'].to(device, non_blocking=True)
        batch['labels'] = batch['labels'].to(device, non_blocking=True)
        if 'images' in batch:
            batch['images'] = batch['images'].to(device, non_blocking=True)
            batch['image_indices'] = batch['image_indices'].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    total_confidence = 0

    with torch.no_grad():
        for batch in dataloader:
            batch['text'] = batch['text'].to(device, non_blocking=True)
            batch['features'] = batch['features'].to(device, non_blocking=True)
            batch['labels'] = batch['labels'].to(device, non_blocking=True)
            if 'images' in batch:
                batch['images'] = batch['images'].to(device, non_blocking=True)
                batch['image_indices'] = batch['image_indices'].to(device, non_blocking=True)

            outputs = model(batch)
            loss = criterion(outputs, batch['labels'])
            total_loss += loss.item()
            
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            total_confidence += confidence.sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    avg_conf = total_confidence / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, f1, avg_conf

# --- Optuna Objective ---
def objective(trial, all_messages, X_text_all, X_features_all, train_idx, val_idx, text_input_size, features_input_size, num_workers):
    try:
        # === 3. УМЕНЬШЕНИЕ LR (СУЖЕНИЕ ДИАПАЗОНА) ===
        # Было до 5e-4, теперь до 1.5e-4, чтобы избежать резкого переобучения
        lr = trial.suggest_float('lr', 1e-5, 1.5e-4, log=True)
        
        dropout = trial.suggest_float('dropout', 0.2, 0.6)
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 5.0)
        scheduler_patience = trial.suggest_int('scheduler_patience', 2, 4)

        model = MetaLearner(
            text_input_size=text_input_size,
            features_input_size=features_input_size,
            dropout=dropout
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_ds = MultiModalDataset(
            [all_messages[i] for i in train_idx], X_text_all[train_idx], X_features_all[train_idx], 
            train_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD
        )
        val_ds = MultiModalDataset(
            [all_messages[i] for i in val_idx], X_text_all[val_idx], X_features_all[val_idx], 
            val_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD
        )

        train_labels = [m['label'] for m in train_ds.messages]
        c_counts = Counter(train_labels)
        weights = [1.0 / (c_counts[l] ** 0.5) for l in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, 
            collate_fn=collate_fn, num_workers=num_workers, 
            persistent_workers=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, 
            collate_fn=collate_fn, num_workers=num_workers, 
            persistent_workers=True, pin_memory=True
        )

        criterion = FocalLoss(gamma=2.0).to(device)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=scheduler_patience)
        
        n_epochs = 12 # Даем чуть больше эпох, но контролируем через Early Stopping
        
        best_val_loss = float('inf')
        early_stop_cnt = 0
        EARLY_STOP_PATIENCE = 3 # Если 3 эпохи подряд лосс не падает - стоп

        for epoch in range(n_epochs): 
            try:
                train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)
                val_loss, val_f1, _ = evaluate_model(model, val_loader, criterion, device)
                
                scheduler.step(val_loss)
                trial.report(val_loss, epoch)
                
                # === 2. EARLY STOPPING ВНУТРИ OPTUNA ===
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                
                if early_stop_cnt >= EARLY_STOP_PATIENCE:
                    # Прерываем попытку, так как модель начала переобучаться
                    # Но возвращаем лучший найденный лосс, а не текущий плохой
                    del train_loader, val_loader
                    return best_val_loss
                # =======================================

                if trial.should_prune():
                    del train_loader, val_loader
                    raise optuna.exceptions.TrialPruned()
                    
            except (OSError, RuntimeError) as e:
                if "Too many open files" in str(e) or "errno 24" in str(e).lower():
                    del train_loader, val_loader
                    return float('inf')
                raise e

        del train_loader, val_loader
        return best_val_loss

    except (OSError, RuntimeError) as e:
        if "Too many open files" in str(e) or "errno 24" in str(e).lower():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return float('inf')
        else:
            raise e

# --- Main ---
def run_training():
    print(f"Using device: {device}")
    
    # DB Management
    if os.path.exists(DB_PATH):
        print(f"\n⚠️ Found DB: {DB_PATH}")
        choice = input("Delete old DB? (y/n) [n]: ").strip().lower()
        if choice == 'y':
            try:
                os.remove(DB_PATH)
                print("🗑️ DB deleted.")
            except Exception as e:
                print(f"❌ Error deleting DB: {e}")
    else:
        print(f"🆕 Creating DB: {DB_PATH}")

    # Cache Loading
    if os.path.exists(CACHE_FILE):
        print(f"✅ Loaded cache: {CACHE_FILE}")
        cache_data = joblib.load(CACHE_FILE)
        all_messages = cache_data['all_messages']
        X_text_all = cache_data['X_text_all']
        X_features_all = cache_data['X_features_all']
        text_vectorizer = cache_data['text_vectorizer']
        feature_scaler = cache_data['feature_scaler']
    else:
        print("❌ Cache not found.")
        return 

    labels = [m['label'] for m in all_messages]
    indices = np.arange(len(all_messages))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=42, stratify=[labels[i] for i in train_val_idx])

    num_workers = 10
    print(f"Num workers: {num_workers}")

    N_TRIALS = 100
    print(f"\nStarting Optuna (With Early Stopping & F1 Monitoring)...")
    
    study = optuna.create_study(
        study_name="multimodal_optimization_v4", 
        storage=STORAGE_URL,
        direction='minimize', 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True
    )
    
    class TqdmCallback:
        def __init__(self, total, pbar):
            self.pbar = pbar
        def __call__(self, study, trial):
            self.pbar.update(1)
            try:
                val = study.best_value
                self.pbar.set_postfix({"Best Loss": f"{val:.5f}"})
            except ValueError:
                self.pbar.set_postfix({"Status": "Wait..."})

    with tqdm(total=N_TRIALS) as pbar:
        try:
            study.optimize(
                lambda trial: objective(trial, all_messages, X_text_all, X_features_all, 
                                      train_idx, val_idx, 
                                      text_vectorizer.max_features, X_features_all.shape[1], 
                                      num_workers), 
                n_trials=N_TRIALS,
                callbacks=[TqdmCallback(N_TRIALS, pbar)],
                gc_after_trial=True 
            )
        except KeyboardInterrupt:
            print("\nOptimization stopped by user.")

    best_params = study.best_params
    print(f"\nBest params: {best_params}")
    
    # --- Final Training with Early Stopping ---
    print("\nTraining final model...")
    all_train_idx = np.concatenate([train_idx, val_idx])
    final_train_ds = MultiModalDataset(
        [all_messages[i] for i in all_train_idx], X_text_all[all_train_idx], X_features_all[all_train_idx], 
        train_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD
    )
    test_ds = MultiModalDataset(
        [all_messages[i] for i in test_idx], X_text_all[test_idx], X_features_all[test_idx], 
        val_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD
    )

    final_model = MetaLearner(
        text_vectorizer.max_features, X_features_all.shape[1], dropout=best_params['dropout']
    ).to(device)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    ft_labels = [m['label'] for m in final_train_ds.messages]
    c_counts = Counter(ft_labels)
    weights = [1.0 / (c_counts[l] ** 0.5) for l in ft_labels] 
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(final_train_ds, batch_size=best_params['batch_size'], sampler=sampler, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True, pin_memory=True)

    criterion = FocalLoss(gamma=2.0).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=best_params['scheduler_patience'])

    best_loss = float('inf')
    early_stop_cnt = 0
    EARLY_STOP_PATIENCE_FINAL = 5 # Для финала можно чуть больше терпения
    FINAL_EPOCHS = 40 # Ставим много, но Early Stopping остановит раньше
    
    for epoch in range(FINAL_EPOCHS): 
        train_loss = train_epoch(final_model, train_loader, criterion, optimizer, device, best_params['max_grad_norm'])
        val_loss, val_f1, val_conf = evaluate_model(final_model, test_loader, criterion, device)
        scheduler.step(val_loss)
        
        # === 1. ВЫВОД F1 В КОНСОЛЬ ===
        print(f"Epoch {epoch+1}/{FINAL_EPOCHS} | Loss: {val_loss:.4f} | Test F1: {val_f1:.4f} | Avg Conf: {val_conf:.2f}")
        
        # === 2. EARLY STOPPING В ФИНАЛЕ ===
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_cnt = 0
            print("  -> Saving best model...")
            torch.save(final_model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
            joblib.dump(text_vectorizer, os.path.join(RESULTS_DIR, 'text_vectorizer.pkl'))
            joblib.dump(feature_scaler, os.path.join(RESULTS_DIR, 'feature_scaler.pkl'))
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= EARLY_STOP_PATIENCE_FINAL:
                print(f"🛑 Early stopping triggered. No improvement for {EARLY_STOP_PATIENCE_FINAL} epochs.")
                break

    print("\nDone! Model saved.")

if __name__ == '__main__':
    run_training()