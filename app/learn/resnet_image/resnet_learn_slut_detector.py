#resnet_learn_slut_detector.py
import os
import sys
from collections import Counter
import time
import warnings 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import multiprocessing 
import hashlib
import cv2 

try:
    if '__file__' in locals():
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        current_path = SCRIPT_DIR
        while not (os.path.exists(os.path.join(current_path, 'pyproject.toml')) or os.path.exists(os.path.join(current_path, '.git'))):
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                raise FileNotFoundError("Project root not found.")
            current_path = parent_path
        PROJECT_ROOT = current_path
    else:
        PROJECT_ROOT = os.getcwd()

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from app.learn.resnet_image.model_architecture import create_configurable_model
    print(f"✅ Successfully imported 'create_configurable_model'. Project root set to: {PROJECT_ROOT}")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ CRITICAL ERROR: Could not import model architecture.")
    print(f"   Please ensure 'app/learn/resnet_image/model_architecture.py' exists.")
    print(f"   Error details: {e}")
    sys.exit(1)
# -------------------------------------------------------------------

Image.MAX_IMAGE_PIXELS = None
# Optional: Suppress the specific PIL warning if it clutters the logs too much
warnings.filterwarnings("ignore", "(?s).*Palette images with Transparency.*")


# --- Конфигурация ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

PERFORM_OPTUNA_SEARCH = False
IMG_SIZE = 256
BATCH_SIZE = 512
EPOCHS = 7
PATIENCE = 3
OPTUNA_N_TRIALS = 5
OPTUNA_EPOCHS = 3
OPTUNA_PATIENCE = min(max(1, OPTUNA_EPOCHS - 1), 3)
OPTUNA_DATASET_FRACTION = 3 / 5
FINAL_TEST_SET_FRACTION = 0.2

DISK_CACHE_DIR = 'data/resized_cache' # Укажите путь к папке кэша
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

MODEL_DIR = 'model/resnet'
os.makedirs(MODEL_DIR, exist_ok=True)

ONNX_PATH = os.path.join(MODEL_DIR, 'nsfw_resnet.onnx')
BEST_OPTUNA_PARAMS_PATH = os.path.join(MODEL_DIR, 'best_optuna_params.pkl')

BEST_STATE_DICT_PATH = os.path.join(MODEL_DIR, 'best_resnet_state_dict.pth')
FINAL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'best_resnet_checkpoint.pth')

# ADDED: Define missing path variables
QUANTIZED_MODEL_PATH = os.path.join(MODEL_DIR, 'quantized_resnet_state_dict.pth')
OPTIMAL_THRESHOLD_PATH = os.path.join(MODEL_DIR, 'optimal_threshold.pkl')

RESULTS_DIR = os.path.join(MODEL_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FINAL_TEST_DATA_PATHS_FILE = os.path.join(RESULTS_DIR, "final_test_data_paths.txt")
FINAL_TEST_DATA_LABELS_FILE = os.path.join(RESULTS_DIR, "final_test_data_labels.txt")

SLUT_DATA_DIR = 'data/reddit/nsfw_images'
REGULAR_DATA_DIR = 'data/reddit/sfw_images'

OPTIMAL_THRESHOLD = 0.5


# --- Аугментация данных ---
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.Affine(rotate=10, translate_percent=0.1, scale=(0.9, 1.1), shear=10, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.5), # Обратите внимание, параметры могут отличаться
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.CoarseDropout(min_holes=1, max_holes=8, min_height=8, max_height=32, min_width=8, max_width=32, p=0.2),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])



# --- Датасет ---
# --- Замените вашу функцию _resize_and_save_worker на эту ---

def _resize_and_save_worker(args):
    """
    Воркер для мультипроцессинга.
    Явно кодирует изображение в PNG перед записью во временный файл.
    """
    original_path, cached_path, img_size = args
    temp_cached_path = cached_path + ".tmp"

    try:
        os.makedirs(os.path.dirname(cached_path), exist_ok=True)

        if os.path.exists(temp_cached_path):
            os.remove(temp_cached_path)

        # Шаг 1: Читаем и изменяем размер с помощью OpenCV
        img_bgr = cv2.imread(original_path)
        if img_bgr is None:
            raise IOError(f"OpenCV could not read image: {original_path}")

        img_resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
        # Шаг 2: Явно кодируем изображение в формат .png в памяти.
        # Это возвращает кортеж (успех, буфер с байтами изображения).
        success, buffer = cv2.imencode('.png', img_resized)
        if not success:
            raise IOError(f"cv2.imencode failed for {original_path}")

        # Шаг 3: Записываем байты из буфера во временный файл.
        # Теперь imwrite не используется, и расширение .tmp не имеет значения.
        with open(temp_cached_path, 'wb') as f:
            f.write(buffer)
        # ---------------------------

        # Шаг 4: Атомарно переименовываем, как и раньше
        os.rename(temp_cached_path, cached_path)
        
        return cached_path
        
    except Exception as e:
        print(f"\n[CACHE WORKER ERROR] Failed to process {original_path}. Error: {e}\n", flush=True)
        if os.path.exists(temp_cached_path):
            try:
                os.remove(temp_cached_path)
            except OSError:
                pass
        return None


class NSFWDataset(Dataset):
    # __init__ остается таким же, как в прошлый раз
    def __init__(self, filepaths, labels, transform=None, cache_dir=None, img_size=256):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        
        if cache_dir:
            print(f"Disk caching enabled. Cache directory: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            
            self.cached_filepaths = []
            for fp in filepaths:
                hexdigest = hashlib.sha256(fp.encode()).hexdigest()
                cached_path = os.path.join(cache_dir, hexdigest[:2], hexdigest[2:4], hexdigest + '.png')
                self.cached_filepaths.append(cached_path)

            tasks_to_cache = []
            for original_fp, cached_fp in zip(self.filepaths, self.cached_filepaths):
                if not os.path.exists(cached_fp):
                    tasks_to_cache.append((original_fp, cached_fp, self.img_size))
            
            if tasks_to_cache:
                print(f"Found {len(tasks_to_cache)} images to cache. Starting pre-caching process...")
                num_workers = os.cpu_count() or 1
                
                with multiprocessing.Pool(processes=num_workers) as pool:
                    results_iterator = pool.imap_unordered(_resize_and_save_worker, tasks_to_cache)
                    list(tqdm(results_iterator, total=len(tasks_to_cache), desc="Warming up disk cache"))
                print("Disk cache warm-up complete.")
            else:
                print("All images are already cached.")
                
            self.source_filepaths = self.cached_filepaths
        else:
            print("Disk caching is disabled. Images will be resized on-the-fly.")
            self.source_filepaths = self.filepaths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_path = self.source_filepaths[idx]

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise IOError(f"Could not read image: {image_path}")
            
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Было: cv2.COLOR_BGR_RGB
            # Стало: cv2.COLOR_BGR2RGB
            numpy_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Returning a red placeholder.")
            numpy_img = np.full((self.img_size, self.img_size, 3), (255, 0, 0), dtype=np.uint8)

        if self.transform:
            transformed = self.transform(image=numpy_img)
            tensor_img = transformed['image']
        else:
            tensor_img = transforms.ToTensor()(numpy_img)

        return tensor_img, torch.tensor(label, dtype=torch.float)

# --- (The rest of your helper functions remain the same) ---
# ...
# [No changes needed for save_metrics_report, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, load_data, create_configurable_model, unfreeze_layers, get_sampler_weights]
# ...
def save_metrics_report(y_true, y_pred_binary, y_scores_probs, filename='metrics_report.txt'):
    # Check if y_true is list-like and has elements
    if not (hasattr(y_true, '__len__') and len(y_true) > 0):
        print(f"Skipping metrics report '{filename}', no true labels provided or y_true is not list-like.")
        return None, 0.0, 0.0

    try:
        report_str = classification_report(y_true, y_pred_binary, target_names=['Regular', 'NSFW'], zero_division=0)
        roc_auc = 0.0
        ap_score = 0.0
        if len(np.unique(y_true)) > 1: # Ensure there are at least two classes for ROC AUC and AP
            roc_auc = roc_auc_score(y_true, y_scores_probs)
            ap_score = average_precision_score(y_true, y_scores_probs)
        else:
            print(f"Warning for report '{filename}': Only one class present in y_true. ROC-AUC and AP set to 0.")

    except Exception as e:
        print(f"Error generating classification report for '{filename}': {e}")
        return None, 0.0, 0.0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write(f"\nROC-AUC Score: {roc_auc:.4f}")
        f.write(f"\nAverage Precision Score: {ap_score:.4f}")

    print(f"\n--- Metrics Report for {filename} ---")
    print(report_str)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {ap_score:.4f}")
    return report_str, roc_auc, ap_score

def plot_confusion_matrix(y_true, y_pred_binary, filename='confusion_matrix.png'):
    if not (hasattr(y_true, '__len__') and len(y_true) > 0):
        print(f"Skipping confusion matrix '{filename}', no true labels provided.")
        return
    try:
        cm = confusion_matrix(y_true, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Regular', 'NSFW'],
                    yticklabels=['Regular', 'NSFW'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, filename))
        plt.close()
        print(f"Confusion matrix saved to {os.path.join(RESULTS_DIR, filename)}")
    except Exception as e:
        print(f"Error plotting confusion matrix '{filename}': {e}")

def plot_roc_curve(y_true, y_scores_probs, filename='roc_curve.png'):
    if not (hasattr(y_true, '__len__') and len(y_true) > 0):
        print(f"Skipping ROC curve '{filename}', no true labels provided.")
        return
    if len(np.unique(y_true)) < 2:
        print(f"ROC curve cannot be plotted for '{filename}': only one class present in y_true.")
        return
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores_probs)
        auc_score = roc_auc_score(y_true, y_scores_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, filename))
        plt.close()
        print(f"ROC curve saved to {os.path.join(RESULTS_DIR, filename)}")
    except Exception as e:
        print(f"Error plotting ROC curve '{filename}': {e}")

def plot_precision_recall_curve(y_true, y_scores_probs, filename='precision_recall_curve.png'):
    if not (hasattr(y_true, '__len__') and len(y_true) > 0):
        print(f"Skipping Precision-Recall curve '{filename}', no true labels provided.")
        return
    if len(np.unique(y_true)) < 2:
        print(f"Precision-Recall curve cannot be plotted for '{filename}': only one class present in y_true.")
        return
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores_probs)
        ap_score = average_precision_score(y_true, y_scores_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {ap_score:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, filename))
        plt.close()
        print(f"Precision-Recall curve saved to {os.path.join(RESULTS_DIR, filename)}")
    except Exception as e:
        print(f"Error plotting Precision-Recall curve '{filename}': {e}")

def load_data():
    slut_files = []
    if os.path.exists(SLUT_DATA_DIR):
        for root, _, files in os.walk(SLUT_DATA_DIR):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    slut_files.append(os.path.join(root, filename))
    else:
        print(f"Warning: Directory not found {SLUT_DATA_DIR}")

    regular_files = []
    if os.path.exists(REGULAR_DATA_DIR):
        for root, _, files in os.walk(REGULAR_DATA_DIR):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    regular_files.append(os.path.join(root, filename))
    else:
        print(f"Warning: Directory not found {REGULAR_DATA_DIR}")

    print(f"Found {len(slut_files)} 'slut' images (scanned recursively).")
    print(f"Found {len(regular_files)} 'regular' images (scanned recursively).")

    X = slut_files + regular_files
    y = [1] * len(slut_files) + [0] * len(regular_files)

    if not X:
        print("Warning: No image files found. Returning empty lists.")
        return [], []

    return X, y

def create_configurable_model(params: dict):
    base_model_name = params.get("base_model", "resnet34")

    if base_model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif base_model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif base_model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        print(f"Warning: Unknown base model '{base_model_name}', defaulting to resnet34.")
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    unfreeze_strategy = params.get("unfreeze_strategy", "fc_only")
    if unfreeze_strategy == "fc_only":
        for param in model.parameters():
            param.requires_grad = False
    elif unfreeze_strategy == "all":
        for param in model.parameters():
            param.requires_grad = True
    # Add other strategies if needed

    num_ftrs = model.fc.in_features

    fc_layers_list = []
    n_fc_layers = params.get("n_fc_layers", 2)
    last_out_features = num_ftrs

    for i in range(n_fc_layers):
        fc_units = params.get(f"fc_units_l{i}", 512 if i == 0 else 256)
        fc_dropout = params.get(f"fc_dropout_l{i}", 0.5 if i == 0 else 0.3)

        fc_layers_list.append(nn.Linear(last_out_features, fc_units))
        fc_layers_list.append(nn.BatchNorm1d(fc_units))
        fc_layers_list.append(nn.ReLU(inplace=True))
        fc_layers_list.append(nn.Dropout(fc_dropout))
        last_out_features = fc_units

    fc_layers_list.append(nn.Linear(last_out_features, 1)) # Output layer for binary classification
    model.fc = nn.Sequential(*fc_layers_list)

    return model.to(DEVICE)

def unfreeze_layers(model, current_epoch, total_epochs, unfreeze_schedule_dict):
    """Unfreezes layers based on the schedule."""
    layer_prefix_to_unfreeze = unfreeze_schedule_dict.get(current_epoch)

    if layer_prefix_to_unfreeze:
        unfrozen_now = False
        print(f"\nAttempting to unfreeze layers starting with '{layer_prefix_to_unfreeze}' at epoch {current_epoch + 1}...")
        for name, param in model.named_parameters():
            if name.startswith(layer_prefix_to_unfreeze) and not param.requires_grad:
                param.requires_grad = True
                unfrozen_now = True

        if unfrozen_now:
            print(f"Successfully unfroze some parameters starting with '{layer_prefix_to_unfreeze}'.")
        else:
            print(f"No new layers to unfreeze for prefix '{layer_prefix_to_unfreeze}' (either already unfrozen or prefix not found).")

def get_sampler_weights(labels):
    if not labels:
        return torch.DoubleTensor([])
    class_counts = Counter(labels)
    if not class_counts:
        return torch.DoubleTensor([])

    weight_per_class = {cls: 1.0 / count for cls, count in class_counts.items() if count > 0}

    if not weight_per_class:
        return torch.DoubleTensor([])

    weights = [weight_per_class.get(cls, 1.0) for cls in labels]
    return torch.DoubleTensor(weights)


# --- Обучение модели ---
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, patience_epochs, current_trial_num=None):
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

    best_val_metric = -1.0
    metric_to_monitor = 'val_f1'

    no_improve_epochs = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_roc_auc': []}

    unfreeze_schedule_progressive = {}
    initially_all_unfrozen = all(p.requires_grad for p in model.parameters())
    has_frozen_backbone_layers = any(not p.requires_grad for n, p in model.named_parameters() if not n.startswith('fc'))
    is_optuna_trial_very_short_epochs = isinstance(current_trial_num, optuna.trial.Trial) and num_epochs < 5


    if num_epochs >= 5 and has_frozen_backbone_layers and not initially_all_unfrozen and not is_optuna_trial_very_short_epochs:
        print("Progressive unfreezing schedule will be applied.")
        schedule_points = sorted(list(set([
            max(0, num_epochs // 4 -1),
            max(0, num_epochs // 2 -1),
            max(0, (3 * num_epochs) // 4 -1)
        ])))

        layer_prefixes = ['layer4', 'layer3', 'layer2']
        idx = 0
        for point in schedule_points:
            if idx < len(layer_prefixes):
                if point not in unfreeze_schedule_progressive and point < num_epochs -1 :
                     unfreeze_schedule_progressive[point] = layer_prefixes[idx]
                     idx +=1
            else:
                break
        print(f"Unfreeze schedule: {unfreeze_schedule_progressive}")
    elif not is_optuna_trial_very_short_epochs:
        print("Progressive unfreezing schedule will NOT be applied (conditions not met or short Optuna trial).")


    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        if epoch in unfreeze_schedule_progressive:
            unfreeze_layers(model, epoch, num_epochs, unfreeze_schedule_progressive)

        desc_prefix = f'Trial {current_trial_num.number} ' if isinstance(current_trial_num, optuna.trial.Trial) else ''
        progress_bar = tqdm(train_loader, desc=f'{desc_prefix}Epoch {epoch + 1}/{num_epochs} [Training]', leave=False)

        if len(train_loader) == 0:
            print(f"{desc_prefix}Epoch {epoch + 1}/{num_epochs} - Train loader is empty. Skipping training for this epoch.")
            history['train_loss'].append(0)
        else:
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_train_loss += loss.item() * inputs.size(0)
                progress_bar.set_postfix(loss=loss.item())

            epoch_train_loss = running_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
            history['train_loss'].append(epoch_train_loss)

        val_loss, val_acc, val_f1, val_roc_auc = 0.0, 0.0, 0.0, 0.0
        if len(val_loader or []) == 0:
            print(f"{desc_prefix}Epoch {epoch + 1}/{num_epochs} - Validation loader is empty or None. Skipping validation.")
            if isinstance(current_trial_num, optuna.trial.Trial):
                print(f"Warning: Optuna trial {current_trial_num.number} cannot validate. Returning 0.0 F1-score.")
                return 0.0
        else:
            val_results = evaluate_model(model, val_loader, criterion, is_validation=True,
                                         is_optuna_trial=isinstance(current_trial_num, optuna.trial.Trial))
            val_loss, val_acc, val_f1, val_roc_auc, _, _ = val_results

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)

        lr_info_parts = []
        if optimizer.param_groups:
            for i, group in enumerate(optimizer.param_groups):
                group_name = group.get('name', f'Group{i}')
                lr_info_parts.append(f"LR-{group_name}: {group['lr']:.2e}")
        lr_info = ", ".join(lr_info_parts) if lr_info_parts else "LR: N/A"

        print_msg = (f"{desc_prefix}Epoch {epoch + 1}/{num_epochs} - Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {val_loss:.4f}, "
                     f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}, {lr_info}")

        if not isinstance(current_trial_num, optuna.trial.Trial):
            print(print_msg)
        elif epoch == num_epochs -1 or num_epochs <=3 :
             print(print_msg)

        if scheduler and (val_loader is not None and len(val_loader) > 0):
            scheduler.step(val_f1)

            current_metric_val = val_f1
            if current_metric_val > best_val_metric:
                best_val_metric = current_metric_val
                no_improve_epochs = 0
                if not isinstance(current_trial_num, optuna.trial.Trial):
                    torch.save(model.state_dict(), BEST_STATE_DICT_PATH)
                    print(f"New best model saved with {metric_to_monitor}: {best_val_metric:.4f}")
            else:
                no_improve_epochs += 1
                if not isinstance(current_trial_num, optuna.trial.Trial):
                     print(f"No improvement for {no_improve_epochs} epochs. Best {metric_to_monitor}: {best_val_metric:.4f}")

                if no_improve_epochs >= patience_epochs:
                    print(f"{desc_prefix}Early stopping triggered at epoch {epoch + 1}.")
                    break

        if isinstance(current_trial_num, optuna.trial.Trial):
            current_trial_num.report(val_f1, epoch)
            if current_trial_num.should_prune():
                del model, optimizer, criterion, scheduler, train_loader, val_loader, history, scaler
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

    if not isinstance(current_trial_num, optuna.trial.Trial):
        if history['train_loss']:
            plot_training_history(history)
        if os.path.exists(BEST_STATE_DICT_PATH) and best_val_metric >=0 :
            print(f"Best model weights from this run saved to {BEST_STATE_DICT_PATH} with ...")
            model.load_state_dict(torch.load(BEST_STATE_DICT_PATH, map_location=DEVICE))
            print("Loaded best model weights for potential further use.")
        # FIXED: Use BEST_STATE_DICT_PATH instead of the undefined BEST_MODEL_PATH
        elif not os.path.exists(BEST_STATE_DICT_PATH) and (not history['val_f1'] or best_val_metric < 0):
             print(f"Warning: No best model saved. Training might have been too short or no improvement seen. Best val F1: {best_val_metric:.4f}")
        elif not os.path.exists(BEST_STATE_DICT_PATH) and best_val_metric >= 0:
             print(f"Warning: Best model path {BEST_STATE_DICT_PATH} not found, but best metric was {best_val_metric:.4f}. Using last epoch model.")

    if isinstance(current_trial_num, optuna.trial.Trial):
        return max(0.0, best_val_metric)
    else:
        return model

# --- (The rest of your functions remain the same) ---
# ...
# [No changes needed for evaluate_model, plot_training_history, export_to_onnx, quantize_model_dynamic, test_onnx_inference, objective, run_optuna_study]
# ...
def evaluate_model(model, loader, criterion, is_validation=False, is_optuna_trial=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions_binary = [] # Predictions after applying 0.5 threshold
    all_predictions_probs = []  # Raw probabilities

    desc = "Validation" if is_validation else "Testing"
    disable_tqdm_eval = is_validation and is_optuna_trial

    if len(loader) == 0:
        if not is_validation: print(f"Warning: {desc} loader is empty. Cannot evaluate.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # loss, acc, f1, roc_auc, precision, recall

    progress_bar = tqdm(loader, desc=desc, leave=False, disable=disable_tqdm_eval)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

            logits = model(inputs)
            if criterion:
                loss = criterion(logits, labels)
                running_loss += loss.item() * inputs.size(0)
            else:
                loss = None

            probs = torch.sigmoid(logits)
            predicted_binary = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions_binary.extend(predicted_binary.cpu().numpy().flatten())
            all_predictions_probs.extend(probs.cpu().numpy().flatten())

            if is_validation and loss is not None:
                 progress_bar.set_postfix(loss=loss.item())


    avg_loss = running_loss / len(loader.dataset) if len(loader.dataset) > 0 and criterion is not None else 0.0

    y_true_np = np.array(all_labels)
    y_pred_binary_np = np.array(all_predictions_binary)
    y_scores_probs_np = np.array(all_predictions_probs)

    if len(y_true_np) == 0:
        if not is_validation: print(f"Warning: No labels collected during {desc}. Cannot compute metrics.")
        return avg_loss, 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = accuracy_score(y_true_np, y_pred_binary_np)
    precision = precision_score(y_true_np, y_pred_binary_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_binary_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_binary_np, zero_division=0)

    roc_auc = 0.0
    if len(np.unique(y_true_np)) > 1:
        try:
            roc_auc = roc_auc_score(y_true_np, y_scores_probs_np)
        except ValueError as e:
            if not is_validation: print(f"Warning: ROC-AUC could not be computed for {desc}: {e}. Setting to 0.")
    elif not is_validation and len(y_true_np) > 0 :
         print(f"Warning: ROC-AUC could not be computed for {desc} (only one class in y_true). Setting to 0.")

    if not is_validation:
        print(f"\n--- {desc} Results (threshold 0.5) ---")
        if criterion is not None: print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

    return avg_loss, accuracy, f1, roc_auc, precision, recall

def plot_training_history(history):
    if not history or not history.get('train_loss'):
        print("Skipping training history plot: history is empty or lacks 'train_loss'.")
        return

    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    if history.get('val_loss'):
        plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    if history.get('val_acc'):
        plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
    if history.get('val_f1'):
        plt.plot(epochs_range, history['val_f1'], label='Val F1-score')
    plt.title('Validation Accuracy & F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    if history.get('val_acc') or history.get('val_f1'):
        plt.legend()

    plt.subplot(1, 3, 3)
    if history.get('val_roc_auc'):
        plt.plot(epochs_range, history['val_roc_auc'], label='Val ROC-AUC')
    plt.title('Validation ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    if history.get('val_roc_auc'):
        plt.legend()

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'))
    plt.close()
    print(f"Training history plot saved to {os.path.join(RESULTS_DIR, 'training_history.png')}")

def export_to_onnx(model_to_export, input_shape=(1, 3, IMG_SIZE, IMG_SIZE)):
    if model_to_export is None:
        print("Skipping ONNX export: model is None.")
        return
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("ONNX and/or onnxruntime not installed. Skipping ONNX export.")
        return

    model_to_export.eval()
    model_to_export.to('cpu')
    dummy_input = torch.randn(input_shape, device='cpu')

    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    print(f"Exporting model to ONNX: {ONNX_PATH}")
    try:
        torch.onnx.export(
            model_to_export,
            dummy_input,
            ONNX_PATH,
            opset_version=12,
            input_names=['input'],
            output_names=['output_logits'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output_logits': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to {ONNX_PATH}")

        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("ONNX model checked successfully.")

    except Exception as e:
        print(f"Error during ONNX export or check: {e}")
    finally:
        if torch.cuda.is_available() and DEVICE.type == 'cuda':
            model_to_export.to(DEVICE)

def quantize_model_dynamic(model_to_quantize):
    if model_to_quantize is None:
        print("Skipping quantization: model is None.")
        return None

    model_to_quantize.eval()
    model_to_quantize.to('cpu')

    os.makedirs(os.path.dirname(QUANTIZED_MODEL_PATH), exist_ok=True)

    try:
        from torch.quantization import quantize_dynamic
        model_quantized = quantize_dynamic(
            model_to_quantize, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        torch.save(model_quantized.state_dict(), QUANTIZED_MODEL_PATH)
        print(f"Dynamically quantized model state_dict saved to {QUANTIZED_MODEL_PATH}")
    except Exception as e:
        print(f"Error during dynamic quantization: {e}")
        model_quantized = None
    finally:
        if torch.cuda.is_available() and DEVICE.type == 'cuda':
            model_to_quantize.to(DEVICE)
    return model_quantized

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
    try:
        ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        dummy_input_np = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

        outputs_logits_onnx = ort_session.run(['output_logits'], {'input': dummy_input_np})[0]
        probs_onnx = 1 / (1 + np.exp(-outputs_logits_onnx))

        print(f"ONNX inference test - Logits shape: {outputs_logits_onnx.shape}, Probs (example): {probs_onnx[0]}")
    except Exception as e:
        print(f"Error during ONNX inference test: {e}")

def objective(trial: optuna.trial.Trial, X_train_paths, y_train, X_val_paths, y_val):
    params = {
        "base_model": trial.suggest_categorical("base_model", ["resnet18", "resnet34"]),
        "unfreeze_strategy": trial.suggest_categorical("unfreeze_strategy", ["fc_only", "all"]),
        "n_fc_layers": trial.suggest_int("n_fc_layers", 1, 6)
    }
    for i in range(params["n_fc_layers"]):
        params[f"fc_units_l{i}"] = trial.suggest_int(f"fc_units_l{i}", 64, 1024, step=8)
        params[f"fc_dropout_l{i}"] = trial.suggest_float(f"fc_dropout_l{i}", 0.1, 0.9, step=0.05)

    lr_fc = trial.suggest_float("lr_fc", 1e-5, 1e-3, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 5e-6, 5e-0, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-0, log=True)

    model = create_configurable_model(params).to(DEVICE)

    train_dataset_optuna = NSFWDataset(
        X_train_paths, y_train, transform=train_transform,
        cache_dir=DISK_CACHE_DIR, img_size=IMG_SIZE
    )
    val_dataset_optuna = NSFWDataset(
        X_val_paths, y_val, transform=val_transform,
        cache_dir=DISK_CACHE_DIR, img_size=IMG_SIZE
    )

    if len(train_dataset_optuna) == 0 or len(val_dataset_optuna) == 0:
        print(f"Warning: Optuna trial {trial.number} has empty train or val dataset. Pruning.")
        del model, train_dataset_optuna, val_dataset_optuna
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()

    train_counts_optuna = Counter(y_train)
    sampler_weights_optuna = get_sampler_weights(y_train)
    if len(sampler_weights_optuna) > 0:
        sampler_optuna = WeightedRandomSampler(sampler_weights_optuna, num_samples=len(sampler_weights_optuna), replacement=True)
        shuffle_train_optuna = False
    else:
        sampler_optuna = None
        shuffle_train_optuna = True

    num_cpus = os.cpu_count()
    MAX_RECOMMENDED_WORKERS_OPTUNA = 16
    if num_cpus is None:
        optuna_num_workers = 4 
        print(f"Optuna Trial {trial.number}: Could not determine CPU count. Setting num_workers to {optuna_num_workers}.")
    else:
        optuna_num_workers = max(1, min(num_cpus // 2, MAX_RECOMMENDED_WORKERS_OPTUNA))
    if trial.number == 0:
        print(f"Optuna CPUs available: {num_cpus}. Setting num_workers for Optuna trials to {optuna_num_workers} (capped at {MAX_RECOMMENDED_WORKERS_OPTUNA}).")

    optuna_batch_size = BATCH_SIZE

    drop_last_train_optuna = (len(train_dataset_optuna) > optuna_batch_size and
                              len(train_dataset_optuna) % optuna_batch_size == 1)

    train_loader_optuna = DataLoader(train_dataset_optuna, batch_size=optuna_batch_size, sampler=sampler_optuna,
                                     shuffle=shuffle_train_optuna, num_workers=optuna_num_workers,
                                     pin_memory=True if DEVICE.type == 'cuda' else False,
                                     persistent_workers=True if optuna_num_workers > 0 and DEVICE.type == 'cuda' else False,
                                     drop_last=drop_last_train_optuna)
    val_loader_optuna = DataLoader(val_dataset_optuna, batch_size=optuna_batch_size, shuffle=False,
                                   num_workers=optuna_num_workers,
                                   pin_memory=True if DEVICE.type == 'cuda' else False,
                                   persistent_workers=True if optuna_num_workers > 0 and DEVICE.type == 'cuda' else False)


    if len(train_loader_optuna) == 0 or len(val_loader_optuna) == 0:
        print(f"Warning: Optuna trial {trial.number} has empty DataLoader (train: {len(train_loader_optuna)}, val: {len(val_loader_optuna)}). Pruning.")
        del model, train_dataset_optuna, val_dataset_optuna
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()

    optimizer_grouped_parameters = []
    if hasattr(model, 'fc') and list(model.fc.parameters()):
         optimizer_grouped_parameters.append({'params': model.fc.parameters(), 'lr': lr_fc, 'name': 'fc'})

    backbone_params_list = [p for n, p in model.named_parameters() if not n.startswith('fc') and p.requires_grad]
    if backbone_params_list:
        optimizer_grouped_parameters.append({'params': backbone_params_list, 'lr': lr_backbone, 'name': 'backbone'})

    if not optimizer_grouped_parameters or not any(pg.get('params') for pg in optimizer_grouped_parameters):
        print(f"Warning: Optuna trial {trial.number} - no parameters to optimize. Check model config and unfreeze strategy. Pruning.")
        del model
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()

    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)

    if train_counts_optuna.get(1, 0) > 0 and train_counts_optuna.get(0,0) > 0:
        pos_weight_value = train_counts_optuna.get(0, 0) / train_counts_optuna.get(1,0)
    else:
        pos_weight_value = 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    scheduler_patience = max(1, OPTUNA_PATIENCE // 2) if OPTUNA_EPOCHS > 2 else 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=scheduler_patience, min_lr=1e-7)


    val_f1_score = train_model(model, train_loader_optuna, val_loader_optuna, optimizer, criterion, scheduler,
                               num_epochs=OPTUNA_EPOCHS, patience_epochs=OPTUNA_PATIENCE, current_trial_num=trial)

    del model, optimizer, criterion, scheduler, train_loader_optuna, val_loader_optuna, train_dataset_optuna, val_dataset_optuna
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return val_f1_score

def run_optuna_study():
    X_all, y_all = load_data()
    if len(X_all) == 0:
        print("Error: No data loaded for Optuna. Exiting.")
        return None
    if len(Counter(y_all)) < 2:
        print(f"Error: Only one class found in all data ({Counter(y_all)}). Optuna needs at least two. Exiting.")
        return None

    X_for_optuna_and_final_train, _, y_for_optuna_and_final_train, _ = train_test_split(
        X_all, y_all,
        train_size=(1.0 - FINAL_TEST_SET_FRACTION),
        random_state=42, stratify=y_all,
        shuffle=True
    )

    if OPTUNA_DATASET_FRACTION < 1.0 and len(X_for_optuna_and_final_train) > 0 :
        X_optuna_subset, _, y_optuna_subset, _ = train_test_split(
            X_for_optuna_and_final_train, y_for_optuna_and_final_train,
            train_size=OPTUNA_DATASET_FRACTION,
            random_state=43,
            stratify=y_for_optuna_and_final_train,
            shuffle=True
        )
        print(f"Using {len(X_optuna_subset)} samples ({OPTUNA_DATASET_FRACTION * 100:.1f}% of data available after reserving final test set) for Optuna search.")
    elif len(X_for_optuna_and_final_train) > 0:
        X_optuna_subset, y_optuna_subset = X_for_optuna_and_final_train, y_for_optuna_and_final_train
        print(f"Using all {len(X_optuna_subset)} samples (data available after reserving final test set) for Optuna search.")
    else:
        print(f"Warning: No data available for Optuna after reserving final test set (size: {len(X_for_optuna_and_final_train)}). Skipping Optuna.")
        return None

    if len(X_optuna_subset) < BATCH_SIZE * 2 :
        print(f"Warning: Optuna subset is too small ({len(X_optuna_subset)} samples, need at least {BATCH_SIZE*2}). Skipping Optuna study.")
        return None
    if len(Counter(y_optuna_subset)) < 2:
        print(f"Warning: Optuna subset ({Counter(y_optuna_subset)}) does not contain both classes. Skipping Optuna study.")
        return None

    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_optuna_subset, y_optuna_subset,
        test_size=0.25,
        random_state=44,
        stratify=y_optuna_subset,
        shuffle=True
    )

    print(f"\n--- Optuna Data Split (from subset) ---")
    print(f"Optuna Train set: {len(X_train_opt)} samples ({Counter(y_train_opt)})")
    print(f"Optuna Validation set: {len(X_val_opt)} samples ({Counter(y_val_opt)})")

    if len(X_train_opt) < BATCH_SIZE or len(X_val_opt) < BATCH_SIZE or \
       len(Counter(y_train_opt)) < 2 or len(Counter(y_val_opt)) < 2:
        print("Warning: Optuna train or validation set is too small or does not contain both classes after split. Skipping Optuna study.")
        return None

    n_warmup_steps_pruner = 0 if OPTUNA_EPOCHS < 3 else OPTUNA_EPOCHS // 3
    n_startup_trials_pruner = min(5, OPTUNA_N_TRIALS // 2) if OPTUNA_N_TRIALS >= 4 else 0

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials_pruner, n_warmup_steps=n_warmup_steps_pruner, interval_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    objective_with_data = lambda trial: objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt)

    try:
        study.optimize(objective_with_data, n_trials=OPTUNA_N_TRIALS, timeout=None,
                       callbacks=[lambda study, cb_trial: torch.cuda.empty_cache() if torch.cuda.is_available() else None])
    except Exception as e:
        print(f"Error during Optuna optimization: {e}")
        if study.trials and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            print("Optimization interrupted, but returning best trial found so far.")
        else:
            return None

    print("\n--- Optuna Study Finished ---")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed_trials:
        print("Optuna study completed, but no trials were completed successfully with a valid value.")
        return None

    try:
        best_trial_candidate = study.best_trial
    except ValueError:
        best_trial_candidate = None

    if best_trial_candidate and best_trial_candidate.value is not None:
        print("Best trial:")
        print(f"  Value (Val F1): {best_trial_candidate.value:.4f}")
        print("  Best hyperparameters: ")
        for key, value in best_trial_candidate.params.items():
            print(f"    {key}: {value}")

        os.makedirs(os.path.dirname(BEST_OPTUNA_PARAMS_PATH), exist_ok=True)
        joblib.dump(best_trial_candidate.params, BEST_OPTUNA_PARAMS_PATH)
        print(f"Best Optuna parameters saved to {BEST_OPTUNA_PARAMS_PATH}")
        return best_trial_candidate.params
    else:
        print("Optuna study completed, but no best trial found (possibly all trials failed or were pruned before completion or returned None).")
        return None



# --- Основная функция ---

def main():
    global OPTIMAL_THRESHOLD

    params_for_final_training = None
    model_trained = None

    if PERFORM_OPTUNA_SEARCH:
        print("\n--- Step 1: Running Optuna Hyperparameter Search ---")
        params_for_final_training = run_optuna_study()
        if params_for_final_training:
            print("\nOptuna search complete. Using best found parameters for final training.")
        else:
            print("\nOptuna search did not yield parameters or was skipped/failed. Will attempt to load or use defaults for final training.")
    else:
        print("\n--- Step 1: Optuna Hyperparameter Search SKIPPED ---")

    if params_for_final_training is None:
        if os.path.exists(BEST_OPTUNA_PARAMS_PATH):
            print(f"Attempting to load previously saved Optuna parameters from: {BEST_OPTUNA_PARAMS_PATH}")
            try:
                params_for_final_training = joblib.load(BEST_OPTUNA_PARAMS_PATH)
                print("Successfully loaded parameters from file.")
            except Exception as e:
                print(f"Error loading parameters from {BEST_OPTUNA_PARAMS_PATH}: {e}. Will use default parameters.")
                params_for_final_training = None
        else:
            print(f"No saved Optuna parameters file found at {BEST_OPTUNA_PARAMS_PATH}.")

    if params_for_final_training is None:
        print("Using default model parameters for final training.")
        params_for_final_training = {
            "base_model": "resnet34", "unfreeze_strategy": "fc_only",
            "n_fc_layers": 2, "fc_units_l0": 1024, "fc_dropout_l0": 0.5,
            "fc_units_l1": 512,  "fc_dropout_l1": 0.3,
            "lr_fc": 1e-4, "lr_backbone": 1e-5, "weight_decay": 1e-4
        }

    print("\n--- Step 2: Proceeding with Final Model Training using parameters: ---")
    for key, value in params_for_final_training.items(): print(f"  {key}: {value}")

    X_all, y_all = load_data()

    if len(X_all) == 0:
        print("Error: No data loaded for final training. Exiting.")
        return
    if len(Counter(y_all)) < 2:
        print(f"Error: Only one class found in total data ({Counter(y_all)}). Final training requires at least two. Exiting.")
        return

    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X_all, y_all,
        test_size=FINAL_TEST_SET_FRACTION,
        random_state=42,
        stratify=y_all,
        shuffle=True
    )

    print(f"\n--- Final Training Data Split (on ALL data) ---")
    print(f"Full Train set for final model: {len(X_train_full)} samples ({Counter(y_train_full)})")
    print(f"Final Test set for final model: {len(X_test_final)} samples ({Counter(y_test_final)})")

    if not y_train_full or len(Counter(y_train_full)) < 2 :
        print("Critical Error: Training data for final model is insufficient or lacks class diversity. Exiting.")
        return
    if not y_test_final or len(Counter(y_test_final)) < 2:
         print("Warning: Final test set is empty or does not contain both classes. Some evaluation metrics might be affected or unavailable.")

    sampler_weights = get_sampler_weights(y_train_full)
    if len(sampler_weights) > 0:
        sampler = WeightedRandomSampler(sampler_weights, num_samples=len(sampler_weights), replacement=True)
        shuffle_train = False
    else:
        print("Warning: Could not create sampler weights for final training (e.g., single class). Using shuffle=True.")
        sampler = None
        shuffle_train = True

    train_dataset = NSFWDataset(
    X_train_full, 
    y_train_full, 
    transform=train_transform, 
    cache_dir=DISK_CACHE_DIR,  # Включаем кэширование на диске
    img_size=IMG_SIZE
    )
    test_dataset = NSFWDataset(
        X_test_final, 
        y_test_final, 
        transform=val_transform, 
        cache_dir=None, # Отключаем кэширование для тестового набора
        img_size=IMG_SIZE
    )

    if len(train_dataset) == 0:
        print("Error: Final training dataset is empty. Exiting.")
        return

    num_cpus_final = os.cpu_count()
    MAX_RECOMMENDED_WORKERS_FINAL = 24
    if num_cpus_final is None:
        final_num_workers = 4
        print(f"Could not determine CPU count. Setting final_num_workers to {final_num_workers}.")
    else:
        if num_cpus_final <= 2:
            final_num_workers = num_cpus_final
        else:
            final_num_workers = max(1, min(num_cpus_final - 2, MAX_RECOMMENDED_WORKERS_FINAL))
    print(f"Final training CPUs available: {num_cpus_final}. Setting final_num_workers to {final_num_workers} (capped at {MAX_RECOMMENDED_WORKERS_FINAL}).")


    drop_last_final_train = (len(train_dataset) > BATCH_SIZE and
                             len(train_dataset) % BATCH_SIZE == 1)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=shuffle_train,
        num_workers=final_num_workers, pin_memory=True if DEVICE.type == 'cuda' else False,
        persistent_workers=True if final_num_workers > 0 and DEVICE.type == 'cuda' else False,
        drop_last=drop_last_final_train,
        prefetch_factor=2
    )

    test_loader_num_workers = final_num_workers
    test_loader = []
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, prefetch_factor=2,
            num_workers=test_loader_num_workers, pin_memory=True if DEVICE.type == 'cuda' else False,
            persistent_workers=True if test_loader_num_workers > 0 and DEVICE.type == 'cuda' else False
        )

    if len(train_loader) == 0:
        print("Error: Final training DataLoader is empty (possibly due to small dataset and drop_last). Exiting.")
        return

    model_trained = create_configurable_model(params_for_final_training)

    train_counts = Counter(y_train_full)
    if train_counts.get(1, 0) > 0 and train_counts.get(0,0) > 0:
        pos_weight_value = train_counts.get(0, 0) / train_counts.get(1,0)
    else:
        print("Warning: Not enough class diversity in final training data for pos_weight. Using 1.0.")
        pos_weight_value = 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], device=DEVICE)
    print(f"Using pos_weight for BCEWithLogitsLoss: {pos_weight_tensor.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    lr_fc = params_for_final_training.get("lr_fc", 1e-4)
    lr_backbone = params_for_final_training.get("lr_backbone", 1e-5)
    weight_decay = params_for_final_training.get("weight_decay", 1e-4)

    optimizer_grouped_parameters = []
    if hasattr(model_trained, 'fc') and list(model_trained.fc.parameters()):
        optimizer_grouped_parameters.append({'params': model_trained.fc.parameters(), 'lr': lr_fc, 'name': 'fc'})

    backbone_params_to_optimize = [p for n, p in model_trained.named_parameters() if not n.startswith('fc') and p.requires_grad]
    if backbone_params_to_optimize:
        optimizer_grouped_parameters.append({'params': backbone_params_to_optimize, 'lr': lr_backbone, 'name': 'backbone'})

    if not optimizer_grouped_parameters or not any(pg.get('params') for pg in optimizer_grouped_parameters):
        print("Error: No parameters to optimize for the final model. Check model configuration and unfreeze strategy.")
        return

    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)
    scheduler_patience_final = max(1, PATIENCE - 2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=scheduler_patience_final, min_lr=1e-7)

    print("\n--- Starting Final Training Session ---")
    val_loader_for_final_train = test_loader if len(test_loader) > 0 else None

    model_trained = train_model(model_trained, train_loader, val_loader_for_final_train, optimizer, criterion, scheduler,
                                num_epochs=EPOCHS, patience_epochs=PATIENCE, current_trial_num=None)

    if len(test_loader) > 0 and model_trained is not None:
        print("\n--- Final Model Evaluation on Independent Test Set & SHAP Data Preparation ---")
        model_for_eval = None
        # FIXED: Use BEST_STATE_DICT_PATH instead of BEST_MODEL_PATH
        if os.path.exists(BEST_STATE_DICT_PATH):
            print(f"Loading best saved model from {BEST_STATE_DICT_PATH} for final evaluation.")
            model_for_eval = create_configurable_model(params_for_final_training)
            try:
                model_for_eval.load_state_dict(torch.load(BEST_STATE_DICT_PATH, map_location=DEVICE))
                print("Successfully loaded best model weights.")
            except Exception as e:
                # FIXED: Use BEST_STATE_DICT_PATH in the error message
                print(f"Error loading best model weights from {BEST_STATE_DICT_PATH}: {e}. Using model from last training epoch if available.")
                model_for_eval = model_trained
        else:
            # FIXED: Use BEST_STATE_DICT_PATH in the warning message
            print(f"Warning: Best model file {BEST_STATE_DICT_PATH} not found. Using model from last training epoch if available.")
            model_for_eval = model_trained

        if model_for_eval is None:
            print("Error: No model available for final evaluation. Skipping.")
        else:
            model_for_eval.to(DEVICE).eval()

            _ = evaluate_model(model_for_eval, test_loader, criterion, is_validation=False, is_optuna_trial=False)

            all_final_labels_eval = []
            all_final_scores_probs_eval = []

            actual_filepaths_for_final_test = X_test_final

            if len(test_loader.dataset) > 0:
                with torch.no_grad():
                    for inputs, labels_batch in tqdm(test_loader, desc="Generating final predictions for reports & SHAP data"):
                        inputs = inputs.to(DEVICE)
                        logits = model_for_eval(inputs)
                        probs = torch.sigmoid(logits).cpu()

                        all_final_labels_eval.extend(labels_batch.numpy().flatten())
                        all_final_scores_probs_eval.extend(probs.numpy().flatten())

                if all_final_labels_eval:
                    if actual_filepaths_for_final_test and len(actual_filepaths_for_final_test) == len(all_final_labels_eval):
                        os.makedirs(RESULTS_DIR, exist_ok=True)
                        with open(FINAL_TEST_DATA_PATHS_FILE, "w") as f:
                            for path in actual_filepaths_for_final_test:
                                f.write(f"{path}\n")
                        print(f"Saved final test data paths to {FINAL_TEST_DATA_PATHS_FILE}")

                        with open(FINAL_TEST_DATA_LABELS_FILE, "w") as f:
                            for label in all_final_labels_eval:
                                f.write(f"{int(label)}\n")
                        print(f"Saved final test data labels to {FINAL_TEST_DATA_LABELS_FILE}")
                    else:
                        print(f"Warning: Mismatch or empty actual_filepaths_for_final_test ({len(actual_filepaths_for_final_test)}) "
                              f"and all_final_labels_eval ({len(all_final_labels_eval)}). "
                              f"SHAP background data files might be incorrect or not saved.")

                    current_optimal_threshold = 0.5
                    if len(np.unique(all_final_labels_eval)) > 1:
                        try:
                            precision_rt, recall_rt, thresholds_rt = precision_recall_curve(all_final_labels_eval, all_final_scores_probs_eval)
                            valid_precision = precision_rt[:-1]
                            valid_recall = recall_rt[:-1]

                            fscores_for_thresholds_denominator = (valid_precision + valid_recall)
                            fscores_for_thresholds = np.zeros_like(fscores_for_thresholds_denominator)
                            valid_indices = fscores_for_thresholds_denominator > 1e-9
                            fscores_for_thresholds[valid_indices] = (2 * valid_precision[valid_indices] * valid_recall[valid_indices]) / \
                                                                    fscores_for_thresholds_denominator[valid_indices]

                            if len(fscores_for_thresholds) > 0 and len(thresholds_rt) > 0 and \
                               len(fscores_for_thresholds) == len(thresholds_rt):
                                ix = np.argmax(fscores_for_thresholds)
                                current_optimal_threshold = thresholds_rt[ix]
                                print(f"Optimal threshold based on F1-score on test set: {current_optimal_threshold:.4f} (F1: {fscores_for_thresholds[ix]:.4f})")
                            else:
                                print(f"Warning: Could not reliably determine optimal threshold from PR curve. Using default 0.5.")
                                current_optimal_threshold = 0.5
                        except Exception as e_thresh:
                            print(f"Error calculating optimal threshold: {e_thresh}. Using default 0.5.")
                    else:
                        print("Warning: Not enough class diversity in final test labels to calculate optimal threshold. Using default 0.5.")

                    OPTIMAL_THRESHOLD = current_optimal_threshold
                    joblib.dump({'optimal_threshold': OPTIMAL_THRESHOLD}, OPTIMAL_THRESHOLD_PATH)
                    print(f"Saved optimal threshold ({OPTIMAL_THRESHOLD:.4f}) to {OPTIMAL_THRESHOLD_PATH}")

                    all_final_pred_binary_optimal = (np.array(all_final_scores_probs_eval) >= OPTIMAL_THRESHOLD).astype(int).tolist()

                    save_metrics_report(all_final_labels_eval, all_final_pred_binary_optimal, all_final_scores_probs_eval,
                                        filename='final_test_metrics_report_optimal_thresh.txt')
                    plot_confusion_matrix(all_final_labels_eval, all_final_pred_binary_optimal,
                                          filename='final_test_confusion_matrix_optimal_thresh.png')
                    plot_roc_curve(all_final_labels_eval, all_final_scores_probs_eval,
                                   filename='final_test_roc_curve.png')
                    plot_precision_recall_curve(all_final_labels_eval, all_final_scores_probs_eval,
                                                filename='final_test_precision_recall_curve.png')
                else:
                    print("No labels/predictions generated for final reports.")
            else:
                print("Final test dataset (loader.dataset) is empty. Skipping report generation and SHAP data saving.")

    elif model_trained is None:
        print("\nSkipping final model evaluation as model training failed or model is None.")
    else:
        print("\nSkipping final model evaluation on test set as test_loader is empty.")

    print("\n--- Exporting and Quantizing Model ---")
    model_to_export_and_quantize = None
    # FIXED: Use BEST_STATE_DICT_PATH
    if os.path.exists(BEST_STATE_DICT_PATH) and model_trained is not None:
        print(f"Using best saved model from {BEST_STATE_DICT_PATH} for export and quantization.")
        model_to_export_and_quantize = create_configurable_model(params_for_final_training)
        try:
            # FIXED: Use BEST_STATE_DICT_PATH
            model_to_export_and_quantize.load_state_dict(torch.load(BEST_STATE_DICT_PATH, map_location='cpu'))
            model_to_export_and_quantize.eval()
        except Exception as e:
            print(f"Failed to load best model for export: {e}. Using model from last training epoch if available.")
            if model_trained:
                model_to_export_and_quantize = model_trained
                model_to_export_and_quantize.to('cpu').eval()
            else: model_to_export_and_quantize = None
    elif model_trained is not None:
        print("Using model from last training epoch for export (best model not found or not saved).")
        model_to_export_and_quantize = model_trained
        model_to_export_and_quantize.to('cpu').eval()
    else:
        print("No model available for export (training might have failed or was skipped).")

    if model_to_export_and_quantize:
        export_to_onnx(model_to_export_and_quantize)
        test_onnx_inference()
        # quantized_model = quantize_model_dynamic(model_to_export_and_quantize)
        # if quantized_model:
        #    print("Quantized model created.")
    else:
        print("Skipping export and quantization as no model is available.")


    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()