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
import joblib
from tqdm import tqdm
import onnx
import onnxruntime as ort
from torch.quantization import quantize_dynamic
import time 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Updated configuration to match new structure
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 256  
EPOCHS = 30
PATIENCE = 5 

# Updated paths to match new structure
MODEL_DIR = 'model/resnet'
os.makedirs(MODEL_DIR, exist_ok=True)

ONNX_PATH = os.path.join(MODEL_DIR, 'nsfw_resnet34.onnx')
QUANTIZED_MODEL_PATH = os.path.join(MODEL_DIR, 'nsfw_resnet34_quantized.pth')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_resnet34.pth')

RESULTS_DIR = 'model/resnet/resnet_results'
os.makedirs(RESULTS_DIR, exist_ok=True)  

# Data paths - assuming data is organized under data/ directory
SLUT_DATA_DIR = 'data/raw/slut'
REGULAR_DATA_DIR = 'data/raw/regular'

# Data augmentation (enhanced)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class NSFWDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None, cache_ram=True):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        self.cache = {}
        self.cache_ram = cache_ram and (len(filepaths) * IMG_SIZE * IMG_SIZE * 3 * 4 < 30e9)  

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if idx in self.cache:
            img = self.cache[idx]
        else:
            img = Image.open(self.filepaths[idx]).convert('RGB')
            if self.cache_ram:
                self.cache[idx] = img
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def save_metrics_report(y_true, y_pred, y_scores, filename='metrics_report.txt'):
    report = classification_report(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nROC-AUC Score: {roc_auc:.4f}")
        f.write(f"\nAverage Precision Score: {ap_score:.4f}")
    
    return report, roc_auc, ap_score

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not-NSFW', 'NSFW'], 
                yticklabels=['Not-NSFW', 'NSFW'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def plot_roc_curve(y_true, y_scores, filename='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_scores):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, filename='precision_recall_curve.png'):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {average_precision_score(y_true, y_scores):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def load_data():
    slut_files = [os.path.join(SLUT_DATA_DIR, f) for f in os.listdir(SLUT_DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    regular_files = [os.path.join(REGULAR_DATA_DIR, f) for f in os.listdir(REGULAR_DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    X = slut_files + regular_files
    y = [1] * len(slut_files) + [0] * len(regular_files)
    return X, y

def build_model():
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    
    # Progressive freezing
    for name, param in model.named_parameters():
        if not name.startswith('layer4') and not name.startswith('fc'):
            param.requires_grad = False
    
    # Enhanced head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model.to(DEVICE)

def unfreeze_layers(model, epoch, total_epochs):
    """Progressive layer unfreezing during training"""
    if epoch == total_epochs // 3:
        print("\nUnfreezing layer3...")
        for name, param in model.named_parameters():
            if name.startswith('layer3'):
                param.requires_grad = True
                
    elif epoch == 2 * total_epochs // 3:
        print("\nUnfreezing layer2...")
        for name, param in model.named_parameters():
            if name.startswith('layer2'):
                param.requires_grad = True

def get_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    weight_per_class = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    weights = [weight_per_class[cls] for cls in labels]
    return torch.DoubleTensor(weights)

def train_model(model, train_loader, val_loader, optimizer, criterion):
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.1)
    best_acc = 0
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Progressive unfreezing
        unfreeze_layers(model, epoch, EPOCHS)
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    plot_training_history(history)
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'))
    plt.close()

def evaluate_model(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    val_loss = running_loss / len(loader)
    val_acc = correct / total
    
    # Additional metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"Validation Metrics - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
          f"ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return val_loss, val_acc

def export_to_onnx(model):
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f"ONNX model saved to {ONNX_PATH}")

def quantize_model(model):
    model_quantized = quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(model_quantized.state_dict(), QUANTIZED_MODEL_PATH)
    print(f"Quantized model saved to {QUANTIZED_MODEL_PATH}")
    return model_quantized

def test_onnx_inference():
    ort_session = ort.InferenceSession(ONNX_PATH)
    dummy_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    outputs = ort_session.run(['output'], {'input': dummy_input})
    print("ONNX inference test:", outputs[0].shape)

def main():
    X, y = load_data()
    
    # Class distribution analysis
    print("Class distribution:", Counter(y))
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create WeightedRandomSampler for class balancing
    weights = get_class_weights(y_train)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_dataset = NSFWDataset(X_train, y_train, train_transform, cache_ram=True)
    test_dataset = NSFWDataset(X_test, y_test, val_transform, cache_ram=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True
    )
    
    model = build_model()
    
    # Add class weights to loss function
    pos_weight = torch.tensor([3.5]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Different learning rates for different layers
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not n.startswith('fc')], 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    print("Starting training...")
    model = train_model(model, train_loader, test_loader, optimizer, criterion)
    
    print("\nFinal model evaluation...")
    y_true, y_pred, y_scores = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).squeeze().cpu().numpy()
            y_scores.extend(outputs)
            y_pred.extend((outputs > 0.5).astype(int))
            y_true.extend(labels.numpy())
    
    # Save metrics and plots
    report, roc_auc, ap_score = save_metrics_report(y_true, y_pred, y_scores)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_scores)
    plot_precision_recall_curve(y_true, y_scores)
    
    print("\nClassification Report:")
    print(report)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {ap_score:.4f}")

    # Export and quantize
    export_to_onnx(model)
    quantized_model = quantize_model(model)
    test_onnx_inference()

if __name__ == "__main__":
    main()