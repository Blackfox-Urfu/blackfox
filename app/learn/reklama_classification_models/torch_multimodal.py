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

# Импортируем наши новые модели
from torch_models import MetaLearner

# --- Конфигурация (без изменений) ---
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


# --- Класс Dataset (без изменений) ---
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
        attachments = msg.get('attachments', [])
        for att in attachments:
            if att.get('type') == 'photo' and att.get('is_valid'):
                try:
                    base_dir = self.base_dir_ad if label == 1 else self.base_dir_non_ad
                    img_path = os.path.join(base_dir, att['path'])
                    if not os.path.exists(img_path): continue
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.image_transform(image)
                    break
                except (FileNotFoundError, IOError, UnidentifiedImageError):
                    continue
        return {'text': text_vector, 'features': features_vector, 'image': image_tensor, 'label': label}

# ИСПРАВЛЕНИЕ 1: Убираем все вызовы .to(device) из collate_fn
def collate_fn(batch):
    texts = torch.tensor(np.array([item['text'] for item in batch]), dtype=torch.float32)
    features = torch.tensor(np.array([item['features'] for item in batch]), dtype=torch.float32)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    images, image_indices = [], []
    for i, item in enumerate(batch):
        if item['image'] is not None:
            images.append(item['image'])
            image_indices.append(i)
    
    # Теперь возвращаем словарь с CPU-тензорами
    collated_batch = {'text': texts, 'features': features, 'labels': labels}
    if images:
        collated_batch['images'] = torch.stack(images)
        collated_batch['image_indices'] = torch.tensor(image_indices, dtype=torch.long)
        
    return collated_batch

# ИСПРАВЛЕНИЕ 2: Добавляем перемещение батча на device внутри функции
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # Перемещаем каждую часть батча на целевое устройство
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ИСПРАВЛЕНИЕ 3: Аналогично для функции evaluate_model
def evaluate_model(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # Перемещаем каждую часть батча на целевое устройство
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            outputs = model(batch)
            loss = criterion(outputs, batch['labels'])
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, f1, report

# --- Основной скрипт (без изменений в логике) ---
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    print(f"Using device: {device}")
    
    print("Loading data...")
    with open(AD_FILE, 'r', encoding='utf-8') as f:
        ad_messages = json.load(f)['messages']
        for msg in ad_messages: msg['label'] = 1
    with open(NON_AD_FILE, 'r', encoding='utf-8') as f:
        non_ad_messages = json.load(f)['messages']
        for msg in non_ad_messages: msg['label'] = 0
    all_messages = ad_messages + non_ad_messages
    print(f"Total messages: {len(all_messages)}. Ads: {len(ad_messages)}, Non-Ads: {len(non_ad_messages)}")
    
    print("Fitting vectorizers and preprocessing all data...")
    all_texts = [msg.get('text_content', '') for msg in all_messages]
    text_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_text_all = text_vectorizer.fit_transform(all_texts).toarray()
    all_features_list = []
    for msg in all_messages:
        features_dict = msg.get('features', {})
        all_features_list.append([
            features_dict.get('text_length', 0), features_dict.get('link_count', 0),
            features_dict.get('mention_count', 0), features_dict.get('hashtag_count', 0),
            features_dict.get('bot_command_count', 0), features_dict.get('custom_emoji_count', 0),
            features_dict.get('emoji_count', 0), float(features_dict.get('has_forward', False)),
            float(features_dict.get('has_inline_buttons', False)), len(msg.get('attachments', []))
        ])
    feature_scaler = StandardScaler()
    X_features_all = feature_scaler.fit_transform(all_features_list)
    
    labels = [m['label'] for m in all_messages]
    indices = np.arange(len(all_messages))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42, stratify=train_val_labels)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = MultiModalDataset([all_messages[i] for i in train_idx], X_text_all[train_idx], X_features_all[train_idx], image_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD)
    val_dataset = MultiModalDataset([all_messages[i] for i in val_idx], X_text_all[val_idx], X_features_all[val_idx], image_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD)
    test_dataset = MultiModalDataset([all_messages[i] for i in test_idx], X_text_all[test_idx], X_features_all[test_idx], image_transform, RAW_DATA_BASE_DIR_AD, RAW_DATA_BASE_DIR_NON_AD)
    train_labels = [labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor([len(train_idx) / class_counts[0], len(train_idx) / class_counts[1]], dtype=torch.float32).to(device)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4
    print(f"Using {NUM_WORKERS} workers for data loading.")
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    print("Initializing model...")
    model = MetaLearner(text_input_size=text_vectorizer.max_features, features_input_size=X_features_all.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)
    print("Start training...")
    best_val_f1 = 0.0
    patience_counter = 0
    max_patience = 7
    num_epochs = 50
    for epoch in range(num_epochs):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Current LR: {current_lr:.6f}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _ = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_f1)
        elapsed_time = time.time() - start_time
        print(f"  [{elapsed_time:.2f}s] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"    -> New best model found! Saving to {RESULTS_DIR}")
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
            joblib.dump(text_vectorizer, os.path.join(RESULTS_DIR, 'text_vectorizer.pkl'))
            joblib.dump(feature_scaler, os.path.join(RESULTS_DIR, 'feature_scaler.pkl'))
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"    -> Early stopping after {max_patience} epochs without improvement.")
                break
    
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'best_model.pth')))
    test_loss, test_f1, test_report = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test F1-score (weighted): {test_f1:.4f}")
    print("\nClassification Report (Test Set):")
    print(json.dumps(test_report, indent=2))