import os
import gc
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Пути к данным
SLUT_DIR = 'data/slut'
REGULAR_DIR = 'data/regular'
HOG_VIS_DIR = 'hog_visualizations'  # Папка для сохранения визуализаций HOG

# Убедимся, что папка для визуализаций существует
os.makedirs(HOG_VIS_DIR, exist_ok=True)

def load_image(img_path, label):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img, label
        return None, None
    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {e}")
        return None, None

def process_single_image(args, params):
    img_path, label = args
    img, label = load_image(img_path, label)
    if img is None:
        return None
    try:
        img_resized = cv2.resize(img, (params['resize'], params['resize']))
        features = hog(
            img_resized,
            orientations=params['orientations'],
            pixels_per_cell=(params['pixels_per_cell'], params['pixels_per_cell']),
            cells_per_block=(params['cells_per_block'], params['cells_per_block']),
            block_norm='L2-Hys'
        )
        return features, label
    except Exception as e:
        print(f"Ошибка при обработке HOG {img_path}: {e}")
        return None

def process_images_in_directory(folder, label, params, max_workers=4):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    features = []
    labels = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, (fp, label), params) for fp in filepaths]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                feat, lbl = result
                features.append(feat)
                labels.append(lbl)
    return np.array(features, dtype=np.float32), np.array(labels)

def optimize_memory_usage():
    gc.collect()

def visualize_hog(img, params, out_filename=None):
    """Визуализирует и сохраняет HOG-признаки"""
    img_resized = cv2.resize(img, (params['resize'], params['resize']))
    _, hog_image = hog(
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

    if out_filename:
        plt.savefig(os.path.join(HOG_VIS_DIR, out_filename), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def objective(trial):
    hog_params = {
        'resize': trial.suggest_categorical('resize', [64, 96]),
        'orientations': trial.suggest_int('orientations', 6, 12),
        'pixels_per_cell': trial.suggest_categorical('pixels_per_cell', [8, 16]),
        'cells_per_block': trial.suggest_categorical('cells_per_block', [2, 3])
    }
    C = trial.suggest_float('C', 0.01, 10.0, log=True)

    try:
        X_slut, y_slut = process_images_in_directory(SLUT_DIR, 1, hog_params)
        optimize_memory_usage()

        X_regular, y_regular = process_images_in_directory(REGULAR_DIR, 0, hog_params)
        optimize_memory_usage()

        if len(X_slut) == 0 or len(X_regular) == 0:
            raise optuna.exceptions.TrialPruned()

        X = np.vstack((X_slut, X_regular))
        y = np.concatenate((y_slut, y_regular))

        del X_slut, y_slut, X_regular, y_regular
        optimize_memory_usage()

        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.5, random_state=42)
        del X, y
        optimize_memory_usage()

        clf = LinearSVC(C=C, max_iter=10000, dual=False)
        scores = cross_val_score(clf, X_sample, y_sample, cv=3, scoring='f1_macro')

        return np.mean(scores)

    except Exception as e:
        print(f"Ошибка в trial: {e}")
        raise optuna.exceptions.TrialPruned()

def train_final_model(best_params):
    X_slut, y_slut = process_images_in_directory(SLUT_DIR, 1, best_params)
    optimize_memory_usage()

    X_regular, y_regular = process_images_in_directory(REGULAR_DIR, 0, best_params)
    optimize_memory_usage()

    X = np.vstack((X_slut, X_regular))
    y = np.concatenate((y_slut, y_regular))

    del X_slut, y_slut, X_regular, y_regular
    optimize_memory_usage()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, y
    optimize_memory_usage()

    clf = LinearSVC(C=best_params['C'], max_iter=10000, dual=False)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n[📊] Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, 'best_hog_model.pkl')
    print("[💾] Модель сохранена в 'best_hog_model.pkl'")
    joblib.dump(best_params, 'hog_params.pkl')
    print("[💾] Параметры HOG сохранены в 'hog_params.pkl'")
    return clf

def show_examples(folder, n=3):
    images = []
    for filename in os.listdir(folder)[:n]:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        if len(images) >= n:
            break

    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images, 1):
        plt.subplot(1, n, i)
        plt.imshow(img, cmap='gray')
        plt.title(os.path.basename(folder))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Примеры изображений из slut:")
    show_examples(SLUT_DIR)

    print("\nПримеры изображений из regular:")
    show_examples(REGULAR_DIR)

    print("\nНачинаем оптимизацию гиперпараметров...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, n_jobs=1)

    print("\n[🏆] Лучшие параметры:")
    print(study.best_params)
    print(f"[🎯] Лучший f1_macro: {study.best_value:.4f}")

    best_params = study.best_params
    best_params['C'] = study.best_params['C']
    clf = train_final_model(best_params)

    # Визуализация HOG
    print("\nВизуализация HOG для примеров:")

    slut_example = next((os.path.join(SLUT_DIR, f) for f in os.listdir(SLUT_DIR)
                         if os.path.isfile(os.path.join(SLUT_DIR, f))), None)
    regular_example = next((os.path.join(REGULAR_DIR, f) for f in os.listdir(REGULAR_DIR)
                            if os.path.isfile(os.path.join(REGULAR_DIR, f))), None)

    if slut_example:
        img = cv2.imread(slut_example, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print("\nПример из slut:")
            visualize_hog(img, best_params, out_filename='slut_example_hog.png')

    if regular_example:
        img = cv2.imread(regular_example, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print("\nПример из regular:")
            visualize_hog(img, best_params, out_filename='regular_example_hog.png')