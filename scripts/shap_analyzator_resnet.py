import os
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from PIL import Image
from tqdm import tqdm
# from sklearn.model_selection import train_test_split # Не используется напрямую

# Добавляем проектный корень в PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Импорт компонентов из обучающего модуля ---
LEARNING_SCRIPT_MODULE_PATH = "app.learn.resnet_learn_slut_detector"

try:
    module_attributes = [
        'DEVICE', 'MODEL_DIR', 'val_transform', 'BEST_MODEL_PATH',
        'BEST_OPTUNA_PARAMS_PATH', 'RESULTS_DIR', 'NSFWDataset',
        'create_configurable_model', 'load_data', 'OPTIMAL_THRESHOLD' # Попытка импортировать OPTIMAL_THRESHOLD
    ]
    module = __import__(LEARNING_SCRIPT_MODULE_PATH, fromlist=module_attributes)

    DEVICE = module.DEVICE
    MODEL_DIR_str = module.MODEL_DIR
    val_transform = module.val_transform
    BEST_MODEL_PATH_str = module.BEST_MODEL_PATH
    BEST_OPTUNA_PARAMS_PATH_str = module.BEST_OPTUNA_PARAMS_PATH
    LEARNING_RESULTS_DIR_str = module.RESULTS_DIR
    NSFWDataset = module.NSFWDataset
    create_configurable_model = module.create_configurable_model
    load_data_from_module = module.load_data
    # Получаем OPTIMAL_THRESHOLD, если он есть, иначе используем 0.5
    OPTIMAL_THRESHOLD = getattr(module, 'OPTIMAL_THRESHOLD', 0.5)

    print(f"Successfully imported components from {LEARNING_SCRIPT_MODULE_PATH}")
    print(f"Using OPTIMAL_THRESHOLD: {OPTIMAL_THRESHOLD}")

    # Приводим пути к абсолютным
    MODEL_DIR = Path(MODEL_DIR_str).resolve()
    if not MODEL_DIR.is_absolute():
        MODEL_DIR = PROJECT_ROOT / MODEL_DIR_str

    BEST_MODEL_PATH = Path(BEST_MODEL_PATH_str).resolve()
    if not BEST_MODEL_PATH.is_absolute():
        BEST_MODEL_PATH = MODEL_DIR / Path(BEST_MODEL_PATH_str).name

    BEST_OPTUNA_PARAMS_PATH = Path(BEST_OPTUNA_PARAMS_PATH_str).resolve()
    if not BEST_OPTUNA_PARAMS_PATH.is_absolute():
        BEST_OPTUNA_PARAMS_PATH = MODEL_DIR / Path(BEST_OPTUNA_PARAMS_PATH_str).name

    LEARNING_RESULTS_DIR = Path(LEARNING_RESULTS_DIR_str).resolve()
    if not LEARNING_RESULTS_DIR.is_absolute():
        LEARNING_RESULTS_DIR = MODEL_DIR / Path(LEARNING_RESULTS_DIR_str).name

except ImportError as e:
    print(f"Error importing from {LEARNING_SCRIPT_MODULE_PATH}: {e}")
    print("Falling back to dummy components. SHAP analysis will likely fail or be inaccurate.")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR = (PROJECT_ROOT / "model" / "optuna_resnet").resolve()
    BEST_MODEL_PATH = (MODEL_DIR / "best_optuna_resnet.pth").resolve()
    BEST_OPTUNA_PARAMS_PATH = (MODEL_DIR / "best_optuna_params.pkl").resolve()
    LEARNING_RESULTS_DIR = (MODEL_DIR / "results").resolve()
    OPTIMAL_THRESHOLD = 0.5 # Fallback threshold
    print(f"Using fallback OPTIMAL_THRESHOLD: {OPTIMAL_THRESHOLD}")


    from torchvision import transforms as T
    IMG_SIZE_SHAP = 224
    val_transform = T.Compose([
        T.Resize((IMG_SIZE_SHAP, IMG_SIZE_SHAP)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def create_configurable_model(params):
        print("WARNING: Using DUMMY create_configurable_model.")
        from torchvision import models as tv_models
        model = tv_models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)
        return model

    class NSFWDataset(torch.utils.data.Dataset):
        def __init__(self, filepaths, labels, transform=None, img_size=224, **kwargs):
            self.filepaths = filepaths
            self.labels = labels
            self.transform = transform
            self.img_size = img_size

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.filepaths[idx]).convert("RGB")
            except Exception:
                print(f"Warning: Could not load image {self.filepaths[idx]}. Using placeholder.")
                img = Image.new("RGB", (self.img_size, self.img_size), color="grey")

            if self.transform:
                img = self.transform(img)
            label_val = self.labels[idx] if self.labels is not None and idx < len(self.labels) else -1
            return img, torch.tensor(label_val, dtype=torch.float)


    def load_data_from_module():
        print("WARNING: Using DUMMY load_data. Please ensure your data paths are correct.")
        IMG_SIZE_SHAP = 224
        dummy_files = [str(PROJECT_ROOT / f"dummy_img_{i}.png") for i in range(100)]
        dummy_labels = [i % 2 for i in range(100)]
        for fp_str in dummy_files:
            fp = Path(fp_str)
            if not fp.exists():
                try:
                    img = Image.new("RGB", (IMG_SIZE_SHAP, IMG_SIZE_SHAP), color="blue")
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    img.save(fp)
                except Exception as e_dummy:
                    print(f"Could not create dummy image {fp}: {e_dummy}")
        return dummy_files, dummy_labels


# --- Конфигурация SHAP анализа ---
SHAP_RESULTS_DIR = (MODEL_DIR / "shap_analysis_results").resolve()
os.makedirs(SHAP_RESULTS_DIR, exist_ok=True)

TARGET_IMAGE_DIR_FOR_ANALYSIS = Path("/home/pesha/projects/blackfox/data/shap_test_resnet").resolve()

SHAP_BACKGROUND_DATA_FILES_LIST_PATH = LEARNING_RESULTS_DIR / "final_test_data_paths.txt"
SHAP_BACKGROUND_DATA_LABELS_LIST_PATH = LEARNING_RESULTS_DIR / "final_test_data_labels.txt"

N_BACKGROUND_SAMPLES = 100

def load_trained_model(model_path, params_path, device):
    model_path_abs = Path(model_path).resolve()
    params_path_abs = Path(params_path).resolve()

    if not model_path_abs.exists():
        print(f"Model file not found: {model_path_abs}")
        return None
    if not params_path_abs.exists():
        print(f"Model parameters file not found: {params_path_abs}")
        return None
    try:
        best_params = joblib.load(params_path_abs)
        print(f"Loaded model parameters from {params_path_abs}")
    except Exception as e:
        print(f"Error loading parameters from {params_path_abs}: {e}")
        return None
    try:
        model = create_configurable_model(best_params)
        model.load_state_dict(torch.load(model_path_abs, map_location=device))
        model.eval().to(device)
        print(f"Model loaded from {model_path_abs} and set to eval mode on {device}.")
        return model
    except Exception as e:
        print(f"Error loading model state_dict from {model_path_abs} or creating model: {e}")
        return None

def get_image_files_from_dir(target_dir: Path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    filepaths = []
    if not target_dir.is_dir():
        print(f"Error: Target directory for analysis does not exist: {target_dir}")
        return filepaths
    print(f"Scanning for images in {target_dir}...")
    for ext in image_extensions:
        filepaths.extend(list(target_dir.rglob(ext)))
    str_filepaths = [str(p) for p in filepaths]
    print(f"Found {len(str_filepaths)} image files.")
    return str_filepaths

def prepare_data_tensors(filepaths, labels, transform, device, dataset_name="data"):
    if not filepaths:
        print(f"Error: No filepaths provided for {dataset_name}.")
        return None, None, None 
    
    effective_labels = labels
    if labels is None or len(labels) != len(filepaths):
        if labels is not None and len(labels) != len(filepaths): # Only print warning if labels were provided but mismatched
            print(f"Warning: Mismatch in length of filepaths ({len(filepaths)}) and labels ({len(labels)}) for {dataset_name}. Using dummy labels.")
        effective_labels = [-1] * len(filepaths) 

    try:
        dataset = NSFWDataset(filepaths, effective_labels, transform=transform)
        if not len(dataset):
            print(f"Error: {dataset_name} dataset is empty after initialization.")
            return None, None, None

        data_list = []
        actual_filepaths_loaded = [] 
        actual_labels_loaded = []

        for i in range(len(dataset)):
            try:
                img_tensor, label_tensor = dataset[i]
                data_list.append(img_tensor)
                actual_filepaths_loaded.append(filepaths[i])
                actual_labels_loaded.append(effective_labels[i])
            except Exception as e_item:
                print(f"Warning: Could not get item {i} ({filepaths[i]}) from {dataset_name} dataset: {e_item}. Skipping.")
        
        if not data_list:
            print(f"Error: No data successfully loaded into {dataset_name}_list.")
            return None, None, None

        data_tensor = torch.stack(data_list).to(device)
        print(f"Successfully prepared {dataset_name} tensor with shape: {data_tensor.shape}")
        return data_tensor, actual_filepaths_loaded, actual_labels_loaded
    except Exception as e:
        print(f"Error creating {dataset_name} dataset or stacking tensors: {e}")
        return None, None, None


def main_shap_analysis():
    print("--- SHAP Analysis ---")
    print(f"Using device: {DEVICE}")

    model = load_trained_model(BEST_MODEL_PATH, BEST_OPTUNA_PARAMS_PATH, DEVICE)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # 1. Подготовка фоновых данных (background data)
    print("\n--- Preparing Background Data ---")
    bg_filepaths_all = []
    bg_labels_all = []

    if SHAP_BACKGROUND_DATA_FILES_LIST_PATH.exists() and SHAP_BACKGROUND_DATA_LABELS_LIST_PATH.exists():
        print(f"Loading background data lists from {SHAP_BACKGROUND_DATA_FILES_LIST_PATH}...")
        try:
            with open(SHAP_BACKGROUND_DATA_FILES_LIST_PATH, "r") as f:
                bg_filepaths_all = [line.strip() for line in f.readlines() if line.strip()]
            with open(SHAP_BACKGROUND_DATA_LABELS_LIST_PATH, "r") as f:
                bg_labels_all = [int(line.strip()) for line in f.readlines() if line.strip()]
            if len(bg_filepaths_all) != len(bg_labels_all):
                print("Warning: Mismatch in lengths of background filepaths and labels.")
            if not bg_filepaths_all: print("Warning: Loaded background filepaths list is empty.")
        except Exception as e:
            print(f"Error loading background data from file lists: {e}. Resetting lists.")
            bg_filepaths_all, bg_labels_all = [], []
    else:
        print(f"Background data list files not found at {SHAP_BACKGROUND_DATA_FILES_LIST_PATH}.")

    if not bg_filepaths_all or not bg_labels_all:
        print("Attempting to load background data using load_data_from_module...")
        try:
            temp_X, temp_y = load_data_from_module()
            bg_filepaths_all, bg_labels_all = temp_X, temp_y
            print(f"Loaded {len(bg_filepaths_all)} filepaths and {len(bg_labels_all)} labels for background.")
        except Exception as e:
            print(f"Could not load background data using load_data_from_module: {e}")
            print("SHAP analysis cannot proceed without background data.")
            return

    if not bg_filepaths_all:
        print("No background data available. Exiting.")
        return

    num_bg_to_sample = min(len(bg_filepaths_all), N_BACKGROUND_SAMPLES)
    if len(bg_filepaths_all) < N_BACKGROUND_SAMPLES:
         print(f"Warning: Available background samples ({len(bg_filepaths_all)}) is less than N_BACKGROUND_SAMPLES ({N_BACKGROUND_SAMPLES}). Using all {num_bg_to_sample} available.")

    indices = np.random.permutation(len(bg_filepaths_all))
    selected_indices = indices[:num_bg_to_sample]
    
    background_fps_selected = [bg_filepaths_all[i] for i in selected_indices]
    background_lbls_selected = [bg_labels_all[i] for i in selected_indices]

    background_data, _, _ = prepare_data_tensors(
        background_fps_selected, 
        background_lbls_selected, 
        val_transform, 
        DEVICE,
        dataset_name="background"
    )
    if background_data is None:
        print("Failed to prepare background data tensor. Exiting.")
        return

    # 2. Подготовка данных для объяснения (explain data) из TARGET_IMAGE_DIR_FOR_ANALYSIS
    print(f"\n--- Preparing Explain Data from {TARGET_IMAGE_DIR_FOR_ANALYSIS} ---")
    explain_filepaths = get_image_files_from_dir(TARGET_IMAGE_DIR_FOR_ANALYSIS)
    if not explain_filepaths:
        print(f"No images found in {TARGET_IMAGE_DIR_FOR_ANALYSIS}. Exiting.")
        return
    
    explain_labels_dummy = [-1] * len(explain_filepaths) # Истинных меток нет, ставим -1

    explain_data, final_explain_fps, _ = prepare_data_tensors( # Нам не нужны final_explain_lbls, т.к. они все -1
        explain_filepaths,
        explain_labels_dummy,
        val_transform,
        DEVICE,
        dataset_name="explain"
    )

    if explain_data is None:
        print("Failed to prepare explain data tensor. Exiting.")
        return
    if not final_explain_fps:
        print(f"No images from {TARGET_IMAGE_DIR_FOR_ANALYSIS} could be loaded. Exiting.")
        return

    print(f"Background data shape: {background_data.shape}")
    print(f"Explain data shape: {explain_data.shape} (from {len(final_explain_fps)} images)")

    # --- Получение предсказаний модели для explain_data ---
    print("\n--- Getting Model Predictions for Explain Data ---")
    model_predictions_classes = []
    model_predictions_probs = []
    with torch.no_grad():
        outputs = model(explain_data) # explain_data уже на DEVICE
        # Предполагаем, что модель выводит логиты (одно значение на семпл для бинарной классификации)
        # outputs должен иметь форму (N_samples, 1)
        probabilities_raw = torch.sigmoid(outputs).cpu().numpy()
        
        # Обработка случая с одним изображением, когда squeeze() может убрать все измерения
        if probabilities_raw.ndim == 2 and probabilities_raw.shape[1] == 1:
            probabilities = probabilities_raw.squeeze(axis=1) # (N_samples,)
        elif probabilities_raw.ndim == 1: # Уже правильная форма
             probabilities = probabilities_raw
        elif probabilities_raw.ndim == 0: # Если был всего 1 семпл и он стал скаляром
            probabilities = np.array([probabilities_raw.item()])
        else: # Неожиданная форма
            print(f"Warning: Probabilities have unexpected shape {probabilities_raw.shape}. Trying to squeeze.")
            try:
                probabilities = probabilities_raw.squeeze()
                if probabilities.ndim == 0: # если стал скаляром
                     probabilities = np.array([probabilities.item()])
                elif probabilities.ndim > 1: # если все еще многомерный
                    raise ValueError(f"Could not reduce probabilities to 1D array. Shape is {probabilities.shape}")
            except Exception as e_prob_shape:
                 print(f"Error reshaping probabilities: {e_prob_shape}. Predictions might be incorrect.")
                 # Создаем заглушки, чтобы не упасть дальше
                 probabilities = np.array([-1.0] * explain_data.shape[0])


        predicted_classes_raw = (probabilities >= OPTIMAL_THRESHOLD).astype(int) # (N_samples,)

    model_predictions_probs = probabilities.tolist()
    model_predictions_classes = predicted_classes_raw.tolist()
    print(f"Predicted classes for {len(model_predictions_classes)} samples obtained.")


    # 3. SHAP Explainer и вычисление SHAP values
    print("\n--- Running SHAP ---")
    print("Creating GradientExplainer...")
    try:
        explainer = shap.GradientExplainer(model, background_data)
        print(f"Calculating SHAP values for {explain_data.shape[0]} samples...")
        shap_values = explainer.shap_values(explain_data)
        print(f"SHAP values calculated. Shape: {np.array(shap_values).shape}")
    except Exception as e:
        print(f"Error during SHAP calculation: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Визуализация и сохранение результатов
    print("\n--- Plotting SHAP Explanations ---")
    mean_norm = np.array([0.485, 0.456, 0.406])
    std_norm = np.array([0.229, 0.224, 0.225])
    
    # Параметры для улучшения качества сохранения
    FIGURE_DPI = 300
    FIGURE_SIZE = (16, 8) # Немного увеличим размер фигуры для лучшей детализации

    for i in tqdm(range(explain_data.shape[0]), desc="Plotting explanations"):
        original_img_tensor = explain_data[i].cpu()
        img_for_plot = original_img_tensor.numpy().transpose(1, 2, 0)
        img_for_plot = std_norm * img_for_plot + mean_norm
        img_for_plot = np.clip(img_for_plot, 0, 1)

        current_filepath = final_explain_fps[i]
        
        # Получаем предсказание для текущего изображения
        predicted_class = model_predictions_classes[i]
        predicted_prob = model_predictions_probs[i]
        pred_label_str = 'NSFW' if predicted_class == 1 else 'Regular'
        
        # Формируем строку с предсказанием
        prediction_info = f"Predicted: {pred_label_str} (Prob: {predicted_prob:.3f})"
        # Истинная метка неизвестна
        true_label_info = "True Label: Unknown"

        plt.figure(figsize=FIGURE_SIZE)
        
        # График оригинального изображения
        plt.subplot(1, 2, 1)
        plt.imshow(img_for_plot, interpolation='bilinear') # Добавим интерполяцию для сглаживания
        title_str = (f"Original: ...{Path(current_filepath).name}\n"
                     f"{true_label_info}\n"
                     f"{prediction_info}")
        plt.title(title_str, fontsize=10)
        plt.axis("off")

        # График SHAP
        plt.subplot(1, 2, 2)
        try:
            current_shap_output_class = shap_values[i]
            if current_shap_output_class.ndim == 4 and current_shap_output_class.shape[-1] == 1:
                shap_map_chw = current_shap_output_class.squeeze(-1)
            elif current_shap_output_class.ndim == 3:
                shap_map_chw = current_shap_output_class
            else:
                raise ValueError(f"SHAP values for sample {i} have unexpected shape {current_shap_output_class.shape}.")
            
            shap_heatmap_hw = np.abs(shap_map_chw).sum(axis=0)
            plt.imshow(shap_heatmap_hw, cmap='viridis') # 'coolwarm' также хороший вариант
            plt.colorbar(label="Sum of abs(SHAP values) across channels")
            plt.title(f"SHAP Explanation", fontsize=10)
            plt.axis("off")

        except Exception as e_plot:
            print(f"Error plotting SHAP for sample {Path(current_filepath).name}: {e_plot}")
            import traceback
            print(traceback.format_exc())
            plt.text(0.5, 0.5, "Error in SHAP plot", ha="center", va="center", transform=plt.gca().transAxes)
            plt.axis("off")

        plt.tight_layout(pad=0.5) # Добавим небольшой отступ
        save_path = SHAP_RESULTS_DIR / f"shap_explanation_{Path(current_filepath).stem}.png"
        try:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        except Exception as e_save:
            print(f"Error saving plot {save_path}: {e_save}")
        plt.close() # Закрываем фигуру для освобождения памяти

    print(f"\nAll explanations saved to {SHAP_RESULTS_DIR}")
    print("--- SHAP Analysis Finished ---")


if __name__ == "__main__":
    main_shap_analysis()