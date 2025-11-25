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

# --- ИСПРАВЛЕНИЕ ИМПОРТА (CRITICAL FIX) ---
# Добавляем текущую директорию в путь поиска модулей, чтобы Python видел соседние файлы
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Этот импорт работает, когда файл запускается как часть пакета (через сервер)
    from .torch_models import MetaLearner
except ImportError:
    # Этот импорт работает, когда файл запускается напрямую через python ...
    from torch_models import MetaLearner
# ------------------------------------------

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Логика извлечения признаков (как на сервере) ---
def extract_features_like_server(text: str, has_image: bool) -> list:
    """
    Эта функция должна БУКВАЛЬНО повторять логику _extract_features из main.py
    """
    if not isinstance(text, str):
        text = ""
    
    text_length = len(text)
    # Точный regex как на сервере
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
        
        # Данные уже подготовлены заранее
        text_vector = self.X_text[idx].astype(np.float32)
        features_vector = self.X_features[idx].astype(np.float32)
        
        image_tensor = None
        # Попытка загрузить изображение, если оно есть в метаданных
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
                except (FileNotFoundError, IOError, UnidentifiedImageError): 
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

# --- Обучение ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # Перенос на устройство
        batch['text'] = batch['text'].to(device)
        batch['features'] = batch['features'].to(device)
        batch['labels'] = batch['labels'].to(device)
        if 'images' in batch:
            batch['images'] = batch['images'].to(device)
            batch['image_indices'] = batch['image_indices'].to(device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch['text'] = batch['text'].to(device)
            batch['features'] = batch['features'].to(device)
            batch['labels'] = batch['labels'].to(device)
            if 'images' in batch:
                batch['images'] = batch['images'].to(device)
                batch['image_indices'] = batch['image_indices'].to(device)

            outputs = model(batch)
            loss = criterion(outputs, batch['labels'])
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, f1, all_labels, all_preds

# --- Optuna Objective ---
def objective(trial, train_dataset, val_dataset, text_input_size, features_input_size, num_workers):
    # Упрощенное пространство поиска, чтобы избежать переобучения
    lr_head = trial.suggest_float('lr_head', 1e-4, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.3, 0.7) # Повышенный dropout
    batch_size = trial.suggest_categorical('batch_size', [32, 64])

    model = MetaLearner(
        text_input_size=text_input_size,
        features_input_size=features_input_size,
        dropout=dropout
    ).to(device)
    
    # Оптимизатор только для головы (backbone заморожен по умолчанию в классе)
    optimizer = optim.AdamW(model.parameters(), lr=lr_head, weight_decay=1e-4)
    
    # Взвешенный сэмплер
    train_labels = [item['label'] for item in train_dataset.messages]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor([1.0/class_counts[0], 1.0/class_counts[1]], dtype=torch.float32)
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Используем persistent_workers=True для ускорения
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True)

    # Label Smoothing помогает от "вечных 100%"
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for epoch in range(5): # Короткие эпохи для Optuna
        train_epoch(model, train_loader, criterion, optimizer, device)
        _, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_f1

# --- Main ---
def run_training():
    try: multiprocessing.set_start_method('spawn')
    except RuntimeError: pass

    print(f"Using device: {device}")
    
    print("Loading data...")
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
    
    # --- ВАЖНО: Подготовка данных ---
    print("Fitting vectorizers and calculating features (Server-Logic compliant)...")
    
    # 1. Текст
    all_texts = [msg.get('text_content', '') or '' for msg in all_messages]
    text_vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=5) # Чуть меньше features для стабильности
    X_text_all = text_vectorizer.fit_transform(all_texts).toarray()
    
    # 2. Признаки (Features) - считаем "на лету", как сервер
    all_features_list = []
    for msg in all_messages:
        txt = msg.get('text_content', '') or ''
        # Проверяем наличие валидного фото
        has_img = False
        for att in msg.get('attachments', []):
            if att.get('type') == 'photo' and att.get('is_valid'):
                has_img = True
                break
        
        feats = extract_features_like_server(txt, has_img)
        all_features_list.append(feats)
        
    feature_scaler = StandardScaler()
    X_features_all = feature_scaler.fit_transform(all_features_list)
    
    print(f"Features shape: {X_features_all.shape}")
    
    # Split
    labels = [m['label'] for m in all_messages]
    indices = np.arange(len(all_messages))
    
    # Train/Val/Test
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=42, stratify=[labels[i] for i in train_val_idx])

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare Datasets
    # Функция для быстрого создания Dataset по индексам
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
    
    num_workers = min(os.cpu_count(), 4) # Не жадничаем с воркерами

    # Optuna
    print("\nStarting Optuna...")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, text_vectorizer.max_features, X_features_all.shape[1], num_workers), 
        n_trials=30 # Достаточно для быстрой настройки
    ) 
    
    best_params = study.best_params
    print(f"\nBest params: {best_params}")

    # Final Training
    print("\nTraining final model...")
    all_train_idx = np.concatenate([train_idx, val_idx])
    final_train_ds = create_ds(all_train_idx)
    test_ds = create_ds(test_idx)
    
    final_model = MetaLearner(text_vectorizer.max_features, X_features_all.shape[1], dropout=best_params['dropout']).to(device)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr_head'], weight_decay=1e-3)
    
    # Sampler для финального обучения
    ft_labels = [m['label'] for m in final_train_ds.messages]
    c_counts = Counter(ft_labels)
    # Смягчаем веса классов, чтобы не перекашивало в одну сторону
    weights = [1.0 / (c_counts[l] ** 0.5) for l in ft_labels] 
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(final_train_ds, batch_size=best_params['batch_size'], sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    # Label Smoothing КЛЮЧЕВОЙ МОМЕНТ для предотвращения 100% уверенности
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
    
    best_f1 = 0.0
    
    for epoch in range(20):
        loss = train_epoch(final_model, train_loader, criterion, optimizer, device)
        _, val_f1, _, _ = evaluate_model(final_model, test_loader, criterion, device)
        
        scheduler.step(val_f1)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Test F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            print("  -> Saving best model...")
            torch.save(final_model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
            joblib.dump(text_vectorizer, os.path.join(RESULTS_DIR, 'text_vectorizer.pkl'))
            joblib.dump(feature_scaler, os.path.join(RESULTS_DIR, 'feature_scaler.pkl'))

    print("\nDone! Model saved.")

if __name__ == '__main__':
    run_training()