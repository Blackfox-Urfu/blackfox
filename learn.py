import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
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

# Конфигурация
DEVICE = torch.device('cuda')
IMG_SIZE = 224
BATCH_SIZE = 512  
EPOCHS = 15
ONNX_PATH = 'model/nsfw_resnet34.onnx'
os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)  

QUANTIZED_MODEL_PATH = 'model/nsfw_resnet34_quantized.pth'
os.makedirs(os.path.dirname(QUANTIZED_MODEL_PATH), exist_ok=True)  

RESULTS_DIR = 'data/result_resnet'
os.makedirs(RESULTS_DIR, exist_ok=True)  


# Аугментация (усиленная)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        self.cache_ram = cache_ram and (len(filepaths) * IMG_SIZE * IMG_SIZE * 3 * 4 < 50e9)  

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
    
    return report, roc_auc, ap_score  # <-- Добавьте эту строку


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
    slut_files = [os.path.join('data/slut', f) for f in os.listdir('data/slut')]
    regular_files = [os.path.join('data/regular', f) for f in os.listdir('data/regular')]
    X = slut_files + regular_files
    y = [1] * len(slut_files) + [0] * len(regular_files)
    return X, y


def build_model():
    model = models.resnet34(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Замораживаем слои
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model.to(DEVICE)

def train_model(model, train_loader, val_loader, optimizer, criterion):
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model/best_resnet34.pth')
    return model

def evaluate_model(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    start_time = time.time()

    with torch.no_grad():
        for count, (inputs, labels) in enumerate(loader):
            print(f'[Eval] Batch {count + 1}/{len(loader)}')
            print(f'        Inputs shape: {inputs.shape}, Labels shape: {labels.shape}')

            inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)

            print(f'        Loss: {loss.item():.4f}, Batch Accuracy: {batch_correct / batch_total:.4f}')

            total += batch_total
            correct += batch_correct

    elapsed = time.time() - start_time
    print(f'Finished test in {elapsed:.2f} seconds')
    print(f'Total Accuracy: {correct / total:.4f}, Avg Loss: {running_loss / len(loader):.4f}')

    return running_loss / len(loader), correct / total

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
    print(f"ONNX-модель сохранена в {ONNX_PATH}")

def quantize_model(model):
    model_quantized = quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(model_quantized.state_dict(), QUANTIZED_MODEL_PATH)
    print(f"Квантованная модель сохранена в {QUANTIZED_MODEL_PATH}")
    return model_quantized

def test_onnx_inference():
    ort_session = ort.InferenceSession(ONNX_PATH)
    dummy_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    outputs = ort_session.run(['output'], {'input': dummy_input})
    print("ONNX inference test:", outputs[0].shape)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = NSFWDataset(X_train, y_train, train_transform, cache_ram=True)
    test_dataset = NSFWDataset(X_test, y_test, val_transform, cache_ram=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model = build_model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Начало обучения...")
    model = train_model(model, train_loader, test_loader, optimizer, criterion)
    
    print("\nТестирование модели...")
    y_true, y_pred, y_scores = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).squeeze().cpu().numpy()
            y_scores.extend(outputs)
            y_pred.extend((outputs > 0.5).astype(int))
            y_true.extend(labels.numpy())
    
    # Сохранение метрик и графиков
    report, roc_auc, ap_score = save_metrics_report(y_true, y_pred, y_scores)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_scores)
    plot_precision_recall_curve(y_true, y_scores)
    
    print("\nClassification Report:")
    print(report)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {ap_score:.4f}")

    # Экспорт и квантование
    export_to_onnx(model)
    quantized_model = quantize_model(model)
    test_onnx_inference()
    
    # Отчет
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = (model(inputs).squeeze() > 0.5).float().cpu().numpy()
            y_pred.extend(outputs)
            y_true.extend(labels.numpy())
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()