import json
import os
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
from sklearn.metrics import classification_report, f1_score
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import joblib
import optuna

# Импортируем наши модели
from .torch_models import MetaLearner # Используем относительный импорт

# --- Конфигурация ---
# Пути определяются относительно запускаемого скрипта
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

# --- Класс Dataset и collate_fn (без изменений) ---
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
                    image_tensor = self.image_transform(image)
                    break
                except (FileNotFoundError, IOError, UnidentifiedImageError): continue
        return {'text': text_vector, 'features': features_vector, 'image': image_tensor, 'label': label}

def collate_fn(batch):
    texts = torch.tensor(np.array([item['text'] for item in batch]), dtype=torch.float32)
    features = torch.tensor(np.array([item['features'] for item in batch]), dtype=torch.float32)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    images, image_indices = [], []
    for i, item in enumerate(batch):
        if item['image'] is not None: images.append(item['image']); image_indices.append(i)
    collated_batch = {'text': texts, 'features': features, 'labels': labels}
    if images:
        collated_batch['images'] = torch.stack(images)
        collated_batch['image_indices'] = torch.tensor(image_indices, dtype=torch.long)
    return collated_batch

# --- Улучшенные функции обучения и валидации ---
def move_batch_to_device(batch, target_device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor): batch[key] = value.to(target_device)
    return batch

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
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
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss = criterion(outputs, batch['labels'])
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, f1, all_labels, all_preds

# --- Интеграция с Optuna ---
def objective(trial, train_dataset, val_dataset, text_input_size, features_input_size, num_workers):
    # Гиперпараметры для поиска
    train_backbone = trial.suggest_categorical('train_backbone', [True, False])
    lr_head = trial.suggest_float('lr_head', 1e-5, 1e-0, log=True)
    lr_backbone = trial.suggest_float('lr_backbone', 1e-6, 1e-0, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.9)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128 , 256])

    model = MetaLearner(
        text_input_size=text_input_size,
        features_input_size=features_input_size,
        dropout=dropout
    ).to(device)
    
    # Устанавливаем, будем ли обучать backbone
    for param in model.image_model.backbone.parameters():
        param.requires_grad = train_backbone

    # Группируем параметры для разных learning rates
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'image_model.backbone' not in n and p.requires_grad], 'lr': lr_head},
        {'params': model.image_model.backbone.parameters(), 'lr': lr_backbone if train_backbone else 0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=1e-4)
    
    # Настройка DataLoader'ов
    train_labels = [item['label'] for item in train_dataset.messages]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor([len(train_labels) / class_counts[0], len(train_labels) / class_counts[1]], dtype=torch.float32).to(device)
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    best_val_f1 = 0.0
    for epoch in range(10): # Ограничимся 10 эпохами для каждого триала Optuna
        train_epoch(model, train_loader, criterion, optimizer, device)
        _, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
    return best_val_f1

# --- Основная функция ---
def run_training():
    try: multiprocessing.set_start_method('spawn')
    except RuntimeError: pass

    print(f"Using device: {device}")
    
    print("Loading data...")
    with open(AD_FILE, 'r', encoding='utf-8') as f: ad_messages = json.load(f)['messages']; [msg.update({'label': 1}) for msg in ad_messages]
    with open(NON_AD_FILE, 'r', encoding='utf-8') as f: non_ad_messages = json.load(f)['messages']; [msg.update({'label': 0}) for msg in non_ad_messages]
    all_messages = ad_messages + non_ad_messages
    print(f"Total messages: {len(all_messages)}. Ads: {len(ad_messages)}, Non-Ads: {len(non_ad_messages)}")
    
    print("Fitting vectorizers and preprocessing data...")
    all_texts = [msg.get('text_content', '') for msg in all_messages]
    text_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_text_all = text_vectorizer.fit_transform(all_texts).toarray()
    
    all_features_list = [[msg.get('features', {}).get(k, 0) for k in ['text_length', 'link_count', 'mention_count', 'hashtag_count']] + [len(msg.get('attachments', []))] for msg in all_messages]
    feature_scaler = StandardScaler()
    X_features_all = feature_scaler.fit_transform(all_features_list)
    
    labels = [m['label'] for m in all_messages]
    indices = np.arange(len(all_messages))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42, stratify=[labels[i] for i in train_val_idx])

    image_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # --- Запускаем Optuna ---
    print("\nStarting hyperparameter optimization with Optuna...")
    train_dataset_opt = MultiModalDataset([all_messages[i] for i in train_idx], X_text_all[train_idx], X_features_all[train_idx], image_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD)
    val_dataset_opt = MultiModalDataset([all_messages[i] for i in val_idx], X_text_all[val_idx], X_features_all[val_idx], image_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD)
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 4

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_dataset_opt, val_dataset_opt, text_vectorizer.max_features, X_features_all.shape[1], num_workers), n_trials=250) 
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")

    # --- Обучаем финальную модель с лучшими параметрами на всех данных (train+val) ---
    print("\nTraining final model with best parameters on combined train+val data...")
    all_train_idx = np.concatenate([train_idx, val_idx])
    final_train_dataset = MultiModalDataset([all_messages[i] for i in all_train_idx], X_text_all[all_train_idx], X_features_all[all_train_idx], image_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD)
    test_dataset = MultiModalDataset([all_messages[i] for i in test_idx], X_text_all[test_idx], X_features_all[test_idx], image_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD)
    
    final_model = MetaLearner(text_vectorizer.max_features, X_features_all.shape[1], dropout=best_params['dropout']).to(device)
    for param in final_model.image_model.backbone.parameters(): param.requires_grad = best_params['train_backbone']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in final_model.named_parameters() if 'image_model.backbone' not in n and p.requires_grad], 'lr': best_params['lr_head']},
        {'params': final_model.image_model.backbone.parameters(), 'lr': best_params['lr_backbone'] if best_params['train_backbone'] else 0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    final_train_labels = [m['label'] for m in final_train_dataset.messages]
    class_counts = Counter(final_train_labels)
    class_weights = torch.tensor([len(final_train_labels) / class_counts[0], len(final_train_labels) / class_counts[1]], dtype=torch.float32).to(device)
    sample_weights = [class_weights[label].item() for label in final_train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(final_train_dataset, batch_size=best_params['batch_size'], sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)
    
    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(50): # Обучаем до 50 эпох с ранней остановкой
        train_loss = train_epoch(final_model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, _, _ = evaluate_model(final_model, test_loader, criterion, device) # Используем test_loader как валидационный
        scheduler.step(val_f1)
        print(f"Epoch {epoch+1}/50 | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"  -> New best model found! Saving to {RESULTS_DIR}")
            torch.save(final_model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
            joblib.dump(text_vectorizer, os.path.join(RESULTS_DIR, 'text_vectorizer.pkl'))
            joblib.dump(feature_scaler, os.path.join(RESULTS_DIR, 'feature_scaler.pkl'))
        else:
            patience_counter += 1
            if patience_counter >= 7: print("  -> Early stopping."); break

    print("\n--- Final Evaluation on Test Set ---")
    final_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'best_model.pth')))
    _, test_f1, test_labels, test_preds = evaluate_model(final_model, test_loader, criterion, device)
    print(f"Final Test F1-score: {test_f1:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(test_labels, test_preds, zero_division=0))

if __name__ == '__main__':
    run_training()