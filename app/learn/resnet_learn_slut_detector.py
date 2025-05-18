import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
# import joblib # Не используется в текущем коде
from tqdm import tqdm
# import onnx # Импортируется позже, если нужно для ONNX
# import onnxruntime as ort # Импортируется позже, если нужно для ONNX
from torch.quantization import quantize_dynamic
# import time # Не используется напрямую
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Конфигурация ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

IMG_SIZE = 224
BATCH_SIZE = 64 # Уменьшил немного, 256 может быть много для некоторых GPU с ResNet34
EPOCHS = 1     
PATIENCE = 5

# Пути
MODEL_DIR = 'model/resnet'
os.makedirs(MODEL_DIR, exist_ok=True)

ONNX_PATH = os.path.join(MODEL_DIR, 'nsfw_resnet34.onnx')
QUANTIZED_MODEL_PATH = os.path.join(MODEL_DIR, 'nsfw_resnet34_quantized.pth')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_resnet34.pth')

RESULTS_DIR = 'model/resnet/resnet_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Пути к данным
SLUT_DATA_DIR = 'data/raw/slut'
REGULAR_DATA_DIR = 'data/raw/regular'

# --- Аугментация данных ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Добавил hue
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), # Усилил affine
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5), # Усилил perspective
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)), # Добавил RandomErasing
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Датасет ---
class NSFWDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None, cache_ram=True):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        self.cache = {}
        # Осторожно с cache_ram: проверяем примерный размер кеша
        # Каждое изображение IMG_SIZE*IMG_SIZE*3 (RGB) * 4 байта (float32 после ToTensor, но PIL Image в RAM будет меньше)
        # Оставим как есть, но при очень больших датасетах может вызвать OOM RAM
        estimated_ram_gb = (len(filepaths) * IMG_SIZE * IMG_SIZE * 3 * 1) / (1024**3) # Примерно, т.к. PIL объекты
        self.cache_ram = cache_ram and (estimated_ram_gb < 20) # Ограничим 20GB для кеша PIL
        if self.cache_ram:
            print(f"RAM caching for PIL Images is enabled. Estimated RAM usage for cache: {estimated_ram_gb:.2f} GB")


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        label = self.labels[idx]
        
        if self.cache_ram and image_path in self.cache:
            img = self.cache[image_path]
        else:
            try:
                img = Image.open(image_path).convert('RGB')
                if self.cache_ram:
                    self.cache[image_path] = img
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Можно вернуть плейсхолдер или пропустить, но лучше обработать при загрузке данных
                # Для простоты, вернем первое валидное изображение (не лучший подход для продакшена)
                placeholder_img_path = self.filepaths[0]
                img = Image.open(placeholder_img_path).convert('RGB')


        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float) # Метка должна быть float для BCEWithLogitsLoss

# --- Вспомогательные функции для метрик и графиков ---
def save_metrics_report(y_true, y_pred_binary, y_scores_probs, filename='metrics_report.txt'):
    report = classification_report(y_true, y_pred_binary, target_names=['Regular', 'NSFW'])
    roc_auc = roc_auc_score(y_true, y_scores_probs)
    ap_score = average_precision_score(y_true, y_scores_probs)
    
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nROC-AUC Score: {roc_auc:.4f}")
        f.write(f"\nAverage Precision Score: {ap_score:.4f}")
    
    print("\n--- Metrics Report ---")
    print(report)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {ap_score:.4f}")
    return report, roc_auc, ap_score

def plot_confusion_matrix(y_true, y_pred_binary, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Regular', 'NSFW'], 
                yticklabels=['Regular', 'NSFW'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Confusion matrix saved to {os.path.join(RESULTS_DIR, filename)}")

def plot_roc_curve(y_true, y_scores_probs, filename='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_scores_probs)
    auc_score = roc_auc_score(y_true, y_scores_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"ROC curve saved to {os.path.join(RESULTS_DIR, filename)}")

def plot_precision_recall_curve(y_true, y_scores_probs, filename='precision_recall_curve.png'):
    precision, recall, _ = precision_recall_curve(y_true, y_scores_probs)
    ap_score = average_precision_score(y_true, y_scores_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {ap_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Precision-Recall curve saved to {os.path.join(RESULTS_DIR, filename)}")

# --- Загрузка данных ---
def load_data():
    slut_files = []
    if os.path.exists(SLUT_DATA_DIR):
        for root, _, files in os.walk(SLUT_DATA_DIR): # Используем os.walk
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    slut_files.append(os.path.join(root, filename))
    else:
        print(f"Warning: Directory not found {SLUT_DATA_DIR}")

    regular_files = []
    if os.path.exists(REGULAR_DATA_DIR):
        for root, _, files in os.walk(REGULAR_DATA_DIR): # Используем os.walk
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    regular_files.append(os.path.join(root, filename))
    else:
        print(f"Warning: Directory not found {REGULAR_DATA_DIR}")
        
    print(f"Found {len(slut_files)} 'slut' images (scanned recursively).")
    print(f"Found {len(regular_files)} 'regular' images (scanned recursively).")

    X = slut_files + regular_files
    y = [1] * len(slut_files) + [0] * len(regular_files) # 1 for NSFW (slut), 0 for SFW (regular)
    
    if not X:
        raise ValueError("No image files found. Please check SLUT_DATA_DIR and REGULAR_DATA_DIR paths.")
        
    return X, y

# --- Построение модели ---
def build_model():
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    
    # Изначально замораживаем все, кроме последних слоев, для fine-tuning
    for name, param in model.named_parameters():
        if not name.startswith('layer4') and not name.startswith('fc'):
            param.requires_grad = False
    
    # Заменяем классификатор (голова модели)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 1)  # ВЫХОД ЛОГИТОВ (без Sigmoid здесь)
    )
    return model.to(DEVICE)

# --- Прогрессивное размораживание ---
def unfreeze_layers(model, epoch, total_epochs, unfreeze_schedule):
    """Progressive layer unfreezing during training"""
    # unfreeze_schedule is a dict like {epoch_num: 'layer_name_prefix'}
    # Example: {5: 'layer3', 10: 'layer2'}
    for unfreeze_epoch, layer_prefix in unfreeze_schedule.items():
        if epoch == unfreeze_epoch:
            print(f"\nUnfreezing {layer_prefix} at epoch {epoch+1}...")
            for name, param in model.named_parameters():
                if name.startswith(layer_prefix):
                    param.requires_grad = True
            print(f"Parameters requiring grad after unfreezing {layer_prefix}:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)


# --- Веса для WeightedRandomSampler ---
def get_sampler_weights(labels):
    class_counts = Counter(labels)
    if not class_counts: return torch.DoubleTensor([]) # Если пустой список
    
    # Вес для каждого класса: 1 / количество_экземпляров_класса
    # Это даст больший вес редким классам
    weight_per_class = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # Для каждого элемента в выборке присваиваем вес его класса
    weights = [weight_per_class[cls] for cls in labels]
    return torch.DoubleTensor(weights)

# --- Обучение модели ---
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler):
    scaler = GradScaler() # Для смешанной точности
    best_val_metric = 0.0  # Используем F1 или ROC-AUC для выбора лучшей модели
    metric_to_monitor = 'val_f1' # или 'val_roc_auc' или 'val_acc'
    
    no_improve_epochs = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_roc_auc': []}

    # Определяем расписание разморозки (пример)
    # Это нужно настроить в зависимости от EPOCHS
    unfreeze_schedule = {
        EPOCHS // 3: 'layer3',
        (2 * EPOCHS) // 3: 'layer2',
        # EPOCHS - (EPOCHS // 4) : 'layer1' # Можно добавить еще позже
    }
    if EPOCHS < 3: # Если эпох мало, разморозка не имеет смысла
        unfreeze_schedule = {}


    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        
        unfreeze_layers(model, epoch, EPOCHS, unfreeze_schedule)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Training]', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1) # labels: [B, 1]
            
            optimizer.zero_grad()
            
            with autocast(): # Включаем смешанную точность
                logits = model(inputs) # Модель возвращает логиты
                loss = criterion(logits, labels) # BCEWithLogitsLoss ожидает логиты
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Валидация
        val_results = evaluate_model(model, val_loader, criterion, is_validation=True)
        val_loss, val_acc, val_f1, val_roc_auc, val_precision, val_recall = val_results

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}, LR: {current_lr:.2e}")

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)
        
        # Обновление scheduler и early stopping
        # ReduceLROnPlateau может работать с разными метриками. Например, val_f1 или val_roc_auc
        # scheduler.step(val_roc_auc) # Пример: оптимизируем по ROC-AUC
        scheduler.step(val_f1) # Или по F1

        # Early stopping logic
        current_metric_val = val_f1 # Метрика для отслеживания улучшения
        if current_metric_val > best_val_metric:
            best_val_metric = current_metric_val
            no_improve_epochs = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved with {metric_to_monitor}: {best_val_metric:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs. Best {metric_to_monitor}: {best_val_metric:.4f}")
            if no_improve_epochs >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1} due to no improvement for {PATIENCE} epochs.")
                break
    
    plot_training_history(history)
    print(f"Best model weights saved to {BEST_MODEL_PATH} with {metric_to_monitor}: {best_val_metric:.4f}")
    
    # Загружаем лучшую модель для дальнейшего использования
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print("Loaded best model weights for final evaluation.")
    else:
        print("Warning: Best model path not found. Using last epoch model.")

    return model

# --- График истории обучения ---
def plot_training_history(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
    plt.plot(epochs_range, history['val_f1'], label='Val F1-score')
    plt.title('Validation Accuracy & F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['val_roc_auc'], label='Val ROC-AUC')
    plt.title('Validation ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'))
    plt.close()
    print(f"Training history plot saved to {os.path.join(RESULTS_DIR, 'training_history.png')}")

# --- Оценка модели ---
def evaluate_model(model, loader, criterion, is_validation=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions_binary = []
    all_predictions_probs = []

    desc = "Validation" if is_validation else "Testing"
    progress_bar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1) # labels: [B, 1]
            
            logits = model(inputs) # Модель возвращает логиты
            loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits) # Преобразуем логиты в вероятности
            predicted_binary = (probs > 0.5).float() # Бинарные предсказания
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions_binary.extend(predicted_binary.cpu().numpy().flatten())
            all_predictions_probs.extend(probs.cpu().numpy().flatten())
            
            if is_validation: # Показываем текущую потерю только на валидации во время трейна
                 progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader.dataset)
    
    # Рассчитываем метрики
    # Убедимся, что есть предсказания для обоих классов, иначе метрики могут выдать ошибку/warning
    # (например, precision_score при отсутствии TP+FP для какого-то класса)
    y_true_np = np.array(all_labels)
    y_pred_binary_np = np.array(all_predictions_binary)
    y_scores_probs_np = np.array(all_predictions_probs)

    accuracy = accuracy_score(y_true_np, y_pred_binary_np)
    
    # Для метрик, чувствительных к отсутствию классов в предсказаниях (например, precision, recall, f1)
    # можно использовать zero_division=0 или 1, или обрабатывать ошибки
    precision = precision_score(y_true_np, y_pred_binary_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_binary_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_binary_np, zero_division=0)
    
    # ROC-AUC требует вероятности
    try:
        roc_auc = roc_auc_score(y_true_np, y_scores_probs_np)
    except ValueError: # Может возникнуть, если только один класс присутствует в y_true
        roc_auc = 0.0 
        print("Warning: ROC-AUC could not be computed (likely only one class in y_true). Setting to 0.")

    if not is_validation: # Печатаем метрики только для финального теста, не для каждой эпохи валидации
        print(f"\n--- {desc} Results ---")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
    return avg_loss, accuracy, f1, roc_auc, precision, recall # Возвращаем все метрики

# --- Экспорт и Квантование ---
def export_to_onnx(model_to_export, input_shape=(1, 3, IMG_SIZE, IMG_SIZE)):
    try:
        import onnx
        import onnxruntime as ort # noqa
    except ImportError:
        print("ONNX and/or onnxruntime not installed. Skipping ONNX export.")
        return

    model_to_export.eval() # Важно перевести модель в режим оценки
    model_to_export.to('cpu') # ONNX экспорт обычно на CPU
    dummy_input = torch.randn(input_shape, device='cpu')
    
    print(f"Exporting model to ONNX: {ONNX_PATH}")
    try:
        torch.onnx.export(
            model_to_export,
            dummy_input,
            ONNX_PATH,
            opset_version=12, # Попробуйте 11, 12 или 13, если есть проблемы
            input_names=['input'],
            output_names=['output_logits'], # Выход теперь логиты
            dynamic_axes={'input': {0: 'batch_size'}, 'output_logits': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to {ONNX_PATH}")

        # Проверка ONNX модели
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("ONNX model checked successfully.")
        
    except Exception as e:
        print(f"Error during ONNX export or check: {e}")
    finally:
        model_to_export.to(DEVICE) # Возвращаем модель на исходное устройство

def quantize_model_dynamic(model_to_quantize):
    model_to_quantize.eval()
    model_to_quantize.to('cpu') # Квантование обычно на CPU
    
    # Динамическое квантование применяется к указанным типам слоев, например, nn.Linear
    model_quantized = quantize_dynamic(
        model_to_quantize, {nn.Linear, nn.Conv2d}, dtype=torch.qint8 # Можно добавить nn.Conv2d
    )
    torch.save(model_quantized.state_dict(), QUANTIZED_MODEL_PATH)
    print(f"Dynamically quantized model state_dict saved to {QUANTIZED_MODEL_PATH}")
    
    model_to_quantize.to(DEVICE) # Возвращаем оригинальную модель на исходное устройство
    return model_quantized # Возвращаем квантованную модель (она на CPU)

def test_onnx_inference():
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping ONNX inference test.")
        return

    if not os.path.exists(ONNX_PATH):
        print(f"ONNX model not found at {ONNX_PATH}. Skipping inference test.")
        return

    print("\nTesting ONNX inference...")
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider']) # или ['CUDAExecutionProvider']
    dummy_input_np = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    
    # ONNX модель выводит логиты
    outputs_logits_onnx = ort_session.run(['output_logits'], {'input': dummy_input_np})[0]
    
    # Для получения вероятностей, применяем sigmoid
    probs_onnx = 1 / (1 + np.exp(-outputs_logits_onnx)) # Sigmoid вручную для numpy
    
    print(f"ONNX inference test - Logits shape: {outputs_logits_onnx.shape}, Probs (example): {probs_onnx[0]}")


# --- Основная функция ---
def main():
    X, y = load_data()
    
    print(f"\n--- Data Status ---")
    print(f"Total images loaded: {len(X)}")
    initial_counts = Counter(y)
    print(f"Initial class distribution: Class 0 (Regular): {initial_counts.get(0,0)}, Class 1 (NSFW): {initial_counts.get(1,0)}")
    
    if len(X) == 0:
        print("Error: No data loaded. Exiting.")
        return
    if len(initial_counts) < 2:
        print(f"Error: Only one class ({len(initial_counts)}) found in the dataset. Need at least two for binary classification. Exiting.")
        return

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n--- Data Split ---")
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    print(f"Train class distribution: Class 0: {train_counts.get(0,0)}, Class 1: {train_counts.get(1,0)}")
    print(f"Test class distribution: Class 0: {test_counts.get(0,0)}, Class 1: {test_counts.get(1,0)}")

    if not train_counts.get(0) or not train_counts.get(1):
        print("Warning: Training set does not contain both classes after split. This might lead to issues.")
    if not test_counts.get(0) or not test_counts.get(1):
         print("Warning: Test set does not contain both classes after split. Some metrics might be uninformative.")


    # WeightedRandomSampler для балансировки классов в батчах на обучении
    sampler_weights = get_sampler_weights(y_train)
    if len(sampler_weights) > 0:
        sampler = WeightedRandomSampler(sampler_weights, num_samples=len(sampler_weights), replacement=True)
        shuffle_train = False # Sampler сам перемешивает
    else:
        print("Warning: Could not create sampler weights (empty y_train or single class). Using shuffle=True for DataLoader.")
        sampler = None
        shuffle_train = True


    train_dataset = NSFWDataset(X_train, y_train, train_transform, cache_ram=True)
    test_dataset = NSFWDataset(X_test, y_test, val_transform, cache_ram=False) # cache_ram=False для теста, если RAM мало
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, # Используем sampler
        shuffle=shuffle_train, # shuffle должен быть False, если используется sampler
        num_workers=max(1, os.cpu_count() // 2), # Безопасное количество воркеров
        pin_memory=True if DEVICE.type == 'cuda' else False # pin_memory только для CUDA
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, # На тесте/валидации не перемешиваем
        num_workers= 2,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    model = build_model()
    
    # --- Определение функции потерь с весами для классов ---
    # pos_weight = num_negative_samples / num_positive_samples
    if train_counts.get(1, 0) > 0: # Избегаем деления на ноль
        pos_weight_value = train_counts.get(0, 0) / train_counts.get(1,0)
    else:
        print("Warning: Class 1 (NSFW) has 0 samples in training set. Using default pos_weight=1.0.")
        pos_weight_value = 1.0
        
    pos_weight_tensor = torch.tensor([pos_weight_value], device=DEVICE)
    print(f"Using pos_weight for BCEWithLogitsLoss: {pos_weight_tensor.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # --- Оптимизатор и планировщик learning rate ---
    # Разные LR для "тела" и "головы" модели
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not n.startswith('fc') and p.requires_grad], 'lr': 1e-5}, # Замороженные слои не попадут
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4) # weight_decay для регуляризации
    
    # Планировщик для уменьшения LR, если метрика не улучшается
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=PATIENCE-2, min_lr=1e-7)


    print("\n--- Starting Training ---")
    model = train_model(model, train_loader, test_loader, optimizer, criterion, scheduler)
    
    print("\n--- Final Model Evaluation on Test Set ---")
    # Загружаем лучшую модель (на всякий случай, если train_model не вернула лучшую)
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH} for final evaluation.")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    else:
        print("Warning: Best model file not found. Evaluating with the last state of the model.")

    # Финальная оценка на тестовом наборе
    # evaluate_model возвращает: avg_loss, accuracy, f1, roc_auc, precision, recall
    final_eval_results = evaluate_model(model, test_loader, criterion, is_validation=False) 
    # avg_loss, accuracy, f1, roc_auc, precision, recall = final_eval_results
    # Эти метрики уже напечатаны внутри evaluate_model при is_validation=False

    # Для генерации отчетов и графиков нужны все предсказания
    model.eval()
    all_final_labels = []
    all_final_pred_binary = []
    all_final_scores_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Generating final predictions"):
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu()
            
            all_final_labels.extend(labels.numpy().flatten()) # labels уже на CPU из DataLoader или нужно .cpu()
            all_final_scores_probs.extend(probs.numpy().flatten())
            all_final_pred_binary.extend((probs > 0.5).float().numpy().flatten())

    # Сохранение метрик и графиков
    save_metrics_report(all_final_labels, all_final_pred_binary, all_final_scores_probs, filename='final_test_metrics_report.txt')
    plot_confusion_matrix(all_final_labels, all_final_pred_binary, filename='final_test_confusion_matrix.png')
    plot_roc_curve(all_final_labels, all_final_scores_probs, filename='final_test_roc_curve.png')
    plot_precision_recall_curve(all_final_labels, all_final_scores_probs, filename='final_test_precision_recall_curve.png')

    # Экспорт и квантование
    print("\n--- Exporting and Quantizing Model ---")
    export_to_onnx(model) # Экспортируем лучшую модель
    test_onnx_inference() # Тестируем ONNX модель
    
    # Квантование (создает новую модель, не изменяет оригинальную `model`)
    # quantized_model_cpu = quantize_model_dynamic(model) 
    # print(f"Quantized model created. Original model device: {next(model.parameters()).device}")
    # print(f"Quantized model device: {next(quantized_model_cpu.parameters()).device}") # Должен быть CPU

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()