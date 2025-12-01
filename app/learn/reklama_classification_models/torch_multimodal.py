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

# --- ИСПРАВЛЕНИЕ ИМПОРТА ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from .torch_models import MetaLearner
except ImportError:
    from torch_models import MetaLearner
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

# Файл кэша для данных
CACHE_FILE = os.path.join(DATA_DIR, "multimodal_training_cache.pkl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def __len__(self): return len(self.messages)
    
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
                    
                    image = Image.open(img_path).convert('RGB')
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
            
    collated_batch = {
        'text': texts, 
        'features': features, 
        'labels': labels
    }
    
    if images:
        collated_batch['images'] = torch.stack(images)
        collated_batch['image_indices'] = torch.tensor(image_indices, dtype=torch.long)
        
    return collated_batch

# --- Обучение (с Gradient Clipping) ---
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
        
        # === Gradient Clipping (защита от взрыва градиентов) ===
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # =======================================================
        
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

# --- Optuna Objective (РАСШИРЕННЫЙ) ---
def objective(trial, train_dataset, val_dataset, text_input_size, features_input_size, num_workers):
    # 1. Learning Rate
    lr = trial.suggest_float('lr', 1e-5, 2e-3, log=True)
    
    # 2. Dropout
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    
    # 3. Batch Size
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # 4. Weight Decay (Регуляризация - "штраф" за сложность)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    # 5. Gradient Clipping Norm (Защита от резких скачков)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 5.0)
    
    # 6. Optimizer Choice (Иногда SGD лучше для картинок)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
    
    # 7. Scheduler settings
    scheduler_patience = trial.suggest_int('scheduler_patience', 1, 3)

    model = MetaLearner(
        text_input_size=text_input_size,
        features_input_size=features_input_size,
        dropout=dropout
    ).to(device)
    
    for param in model.parameters():
        param.requires_grad = True
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # SGD с моментумом (классика для Computer Vision)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    train_labels = [item['label'] for item in train_dataset.messages]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor([1.0/class_counts[0], 1.0/class_counts[1]], dtype=torch.float32)
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        persistent_workers=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        persistent_workers=True,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=scheduler_patience)
    
    for epoch in range(6): # Чуть увеличим, чтобы увидеть динамику
        train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)
        val_loss, val_f1, _ = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

# --- Main ---
def run_training():
    print(f"Using device: {device}")
    
    # Кэширование
    if os.path.exists(CACHE_FILE):
        print(f"✅ Found cache file: {CACHE_FILE}")
        print("Loading data from cache...")
        cache_data = joblib.load(CACHE_FILE)
        all_messages = cache_data['all_messages']
        X_text_all = cache_data['X_text_all']
        X_features_all = cache_data['X_features_all']
        text_vectorizer = cache_data['text_vectorizer']
        feature_scaler = cache_data['feature_scaler']
    else:
        print("❌ Cache not found. Processing raw data...")
        try:
            with open(AD_FILE, 'r', encoding='utf-8') as f: 
                ad_messages = json.load(f)['messages']
                for msg in ad_messages: msg['label'] = 1
                    
            with open(NON_AD_FILE, 'r', encoding='utf-8') as f: 
                non_ad_messages = json.load(f)['messages']
                for msg in non_ad_messages: msg['label'] = 0
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return
        all_messages = ad_messages + non_ad_messages
        
        print("Fitting vectorizer...")
        all_texts = [msg.get('text_content', '') or '' for msg in all_messages]
        text_vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=5)
        X_text_all = text_vectorizer.fit_transform(all_texts).toarray()
        
        print("Extracting features...")
        all_features_list = []
        for msg in all_messages:
            txt = msg.get('text_content', '') or ''
            has_img = False
            for att in msg.get('attachments', []):
                if att.get('type') == 'photo' and att.get('is_valid'):
                    has_img = True
                    break
            feats = extract_features_like_server(txt, has_img)
            all_features_list.append(feats)
            
        feature_scaler = StandardScaler()
        X_features_all = feature_scaler.fit_transform(all_features_list)
        
        print(f"Saving cache to {CACHE_FILE}...")
        joblib.dump({
            'all_messages': all_messages,
            'X_text_all': X_text_all,
            'X_features_all': X_features_all,
            'text_vectorizer': text_vectorizer,
            'feature_scaler': feature_scaler
        }, CACHE_FILE)

    labels = [m['label'] for m in all_messages]
    indices = np.arange(len(all_messages))
    
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=42, stratify=[labels[i] for i in train_val_idx])

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def create_ds(idxs):
        return MultiModalDataset(
            [all_messages[i] for i in idxs], 
            X_text_all[idxs], 
            X_features_all[idxs], 
            image_transform, 
            RAW_DATA_BASE_DIR_AD, 
            RAW_DATA_BASE_DIR_NON_AD
        )

    train_ds = create_ds(train_idx)
    val_ds = create_ds(val_idx)
    
    num_workers = 10 
    print(f"Num workers: {num_workers}")

    print("\nStarting Optuna (Extended Search Space)...")
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    # Увеличим количество попыток, так как параметров стало больше
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, text_vectorizer.max_features, X_features_all.shape[1], num_workers), 
        n_trials=50 
    ) 
    
    best_params = study.best_params
    print(f"\nBest params: {best_params}")
    print(f"Best Loss: {study.best_value}")

    # Final Training
    print("\nTraining final model...")
    all_train_idx = np.concatenate([train_idx, val_idx])
    final_train_ds = create_ds(all_train_idx)
    test_ds = create_ds(test_idx)
    
    final_model = MetaLearner(text_vectorizer.max_features, X_features_all.shape[1], dropout=best_params['dropout']).to(device)
    for param in final_model.parameters(): param.requires_grad = True
        
    if best_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.SGD(final_model.parameters(), lr=best_params['lr'], momentum=0.9, weight_decay=best_params['weight_decay'])
    
    ft_labels = [m['label'] for m in final_train_ds.messages]
    c_counts = Counter(ft_labels)
    weights = [1.0 / (c_counts[l] ** 0.5) for l in ft_labels] 
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(final_train_ds, batch_size=best_params['batch_size'], sampler=sampler, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=best_params['scheduler_patience'])
    
    best_loss = float('inf')
    
    for epoch in range(30): 
        # Не забываем передать max_grad_norm в train_epoch
        train_loss = train_epoch(final_model, train_loader, criterion, optimizer, device, best_params['max_grad_norm'])
        val_loss, val_f1, val_conf = evaluate_model(final_model, test_loader, criterion, device)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} | Loss: {val_loss:.4f} | Test F1: {val_f1:.4f} | Avg Conf: {val_conf:.2f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            print("  -> Saving best model...")
            torch.save(final_model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
            joblib.dump(text_vectorizer, os.path.join(RESULTS_DIR, 'text_vectorizer.pkl'))
            joblib.dump(feature_scaler, os.path.join(RESULTS_DIR, 'feature_scaler.pkl'))

    print("\nDone! Model saved.")

if __name__ == '__main__':
    run_training()