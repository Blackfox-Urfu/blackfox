import os
import cv2
import numpy as np
import optuna
import joblib
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Пути к данным
SLUT_DIR = 'data/slut'
REGULAR_DIR = 'data/regular'

def load_images_from_folder(folder, label):
    """Загружает все изображения из указанной папки и присваивает им метку"""
    features = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    features.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
    return features, labels

def extract_hog_features(images, params):
    """Извлекает HOG-признаки для списка изображений"""
    hog_features = []
    for img in images:
        # Изменение размера
        img_resized = cv2.resize(img, (params['resize'], params['resize']))
        
        # Извлечение HOG-признаков
        features = hog(
            img_resized,
            orientations=params['orientations'],
            pixels_per_cell=(params['pixels_per_cell'], params['pixels_per_cell']),
            cells_per_block=(params['cells_per_block'], params['cells_per_block']),
            block_norm='L2-Hys'
        )
        hog_features.append(features)
    return np.array(hog_features)

def visualize_hog(img, params):
    """Визуализирует HOG-признаки для одного изображения"""
    img_resized = cv2.resize(img, (params['resize'], params['resize']))
    features, hog_image = hog(
        img_resized,
        orientations=params['orientations'],
        pixels_per_cell=(params['pixels_per_cell'], params['pixels_per_cell']),
        cells_per_block=(params['cells_per_block'], params['cells_per_block']),
        block_norm='L2-Hys',
        visualize=True
    )
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def objective(trial):
    """Функция для оптимизации гиперпараметров с помощью Optuna"""
    # Параметры для оптимизации
    hog_params = {
        'resize': trial.suggest_categorical('resize', [64, 96, 128]),
        'orientations': trial.suggest_int('orientations', 6, 12),
        'pixels_per_cell': trial.suggest_categorical('pixels_per_cell', [4, 8, 16]),
        'cells_per_block': trial.suggest_categorical('cells_per_block', [1, 2, 3])
    }
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    
    # Загрузка данных
    slut_images, slut_labels = load_images_from_folder(SLUT_DIR, 1)
    regular_images, regular_labels = load_images_from_folder(REGULAR_DIR, 0)
    
    if not slut_images or not regular_images:
        raise optuna.exceptions.TrialPruned()
    
    # Извлечение признаков
    X_slut = extract_hog_features(slut_images, hog_params)
    X_regular = extract_hog_features(regular_images, hog_params)
    
    X = np.vstack((X_slut, X_regular))
    y = np.array(slut_labels + regular_labels)
    
    # Кросс-валидация
    clf = LinearSVC(C=C, max_iter=10000)
    scores = cross_val_score(clf, X, y, cv=3, scoring='f1_macro')
    
    return np.mean(scores)

def train_final_model(best_params):
    """Обучает финальную модель с лучшими параметрами"""
    # Загрузка данных
    slut_images, slut_labels = load_images_from_folder(SLUT_DIR, 1)
    regular_images, regular_labels = load_images_from_folder(REGULAR_DIR, 0)
    
    # Извлечение признаков
    X_slut = extract_hog_features(slut_images, best_params)
    X_regular = extract_hog_features(regular_images, best_params)
    
    X = np.vstack((X_slut, X_regular))
    y = np.array(slut_labels + regular_labels)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    clf = LinearSVC(C=best_params['C'], max_iter=10000)
    clf.fit(X_train, y_train)
    
    # Оценка
    y_pred = clf.predict(X_test)
    print("\n[📊] Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Сохранение модели
    joblib.dump(clf, 'best_hog_model.pkl')
    print("[💾] Модель сохранена в 'best_hog_model.pkl'")
    
    return clf

def show_examples(folder, n=3):
    """Показывает примеры изображений из папки"""
    images = []
    for filename in os.listdir(folder)[:n]:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images, 1):
        plt.subplot(1, n, i)
        plt.imshow(img, cmap='gray')
        plt.title(os.path.basename(folder))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Показываем примеры изображений
    print("Примеры изображений из slut:")
    show_examples(SLUT_DIR)
    
    print("\nПримеры изображений из regular:")
    show_examples(REGULAR_DIR)
    
    # Оптимизация гиперпараметров
    print("\nНачинаем оптимизацию гиперпараметров...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, n_jobs=-1)
    
    print("\n[🏆] Лучшие параметры:")
    print(study.best_params)
    print(f"[🎯] Лучший f1_macro: {study.best_value:.4f}")
    
    # Обучение финальной модели
    best_params = study.best_params
    clf = train_final_model(best_params)
    
    # Визуализация HOG для примеров
    slut_images, _ = load_images_from_folder(SLUT_DIR, 1)
    regular_images, _ = load_images_from_folder(REGULAR_DIR, 0)
    
    if slut_images:
        print("\nВизуализация HOG для примера из slut:")
        visualize_hog(slut_images[0], best_params)
    
    if regular_images:
        print("\nВизуализация HOG для примера из regular:")
        visualize_hog(regular_images[0], best_params)