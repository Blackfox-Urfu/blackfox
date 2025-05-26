import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from PIL import Image
from tqdm import tqdm # Not used in the final version, but kept from original
from sklearn.model_selection import train_test_split # Not used directly, but kept

# Добавляем проектный корень в PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Импорт компонентов из обучающего модуля ---
LEARNING_SCRIPT_MODULE_PATH = "app.learn.resnet_learn_slut_detector"

try:
    module = __import__(LEARNING_SCRIPT_MODULE_PATH, fromlist=[
        'DEVICE', 'MODEL_DIR', 'val_transform', 'BEST_MODEL_PATH',
        'BEST_OPTUNA_PARAMS_PATH', 'RESULTS_DIR', 'NSFWDataset',
        'create_configurable_model', 'load_data' # Added load_data here
    ])

    DEVICE = module.DEVICE
    MODEL_DIR_str = module.MODEL_DIR
    val_transform = module.val_transform
    BEST_MODEL_PATH_str = module.BEST_MODEL_PATH
    BEST_OPTUNA_PARAMS_PATH_str = module.BEST_OPTUNA_PARAMS_PATH
    LEARNING_RESULTS_DIR_str = module.RESULTS_DIR
    NSFWDataset = module.NSFWDataset
    create_configurable_model = module.create_configurable_model
    load_data_from_module = module.load_data # Renamed to avoid conflict

    print(f"Successfully imported components from {LEARNING_SCRIPT_MODULE_PATH}")

    # Приводим пути к абсолютным
    MODEL_DIR = Path(MODEL_DIR_str).resolve()
    if not MODEL_DIR.is_absolute():
        MODEL_DIR = PROJECT_ROOT / MODEL_DIR_str

    BEST_MODEL_PATH = Path(BEST_MODEL_PATH_str).resolve()
    if not BEST_MODEL_PATH.is_absolute():
        BEST_MODEL_PATH = MODEL_DIR / Path(BEST_MODEL_PATH_str).name # Use name if it's relative to MODEL_DIR

    BEST_OPTUNA_PARAMS_PATH = Path(BEST_OPTUNA_PARAMS_PATH_str).resolve()
    if not BEST_OPTUNA_PARAMS_PATH.is_absolute():
        BEST_OPTUNA_PARAMS_PATH = MODEL_DIR / Path(BEST_OPTUNA_PARAMS_PATH_str).name # Use name

    LEARNING_RESULTS_DIR = Path(LEARNING_RESULTS_DIR_str).resolve()
    if not LEARNING_RESULTS_DIR.is_absolute():
        LEARNING_RESULTS_DIR = MODEL_DIR / Path(LEARNING_RESULTS_DIR_str).name # Use name

except ImportError as e:
    print(f"Error importing from {LEARNING_SCRIPT_MODULE_PATH}: {e}")
    print("Falling back to dummy components. SHAP analysis will likely fail or be inaccurate.")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR = (PROJECT_ROOT / "model" / "optuna_resnet").resolve()
    BEST_MODEL_PATH = (MODEL_DIR / "best_optuna_resnet.pth").resolve()
    BEST_OPTUNA_PARAMS_PATH = (MODEL_DIR / "best_optuna_params.pkl").resolve()
    LEARNING_RESULTS_DIR = (MODEL_DIR / "results").resolve()

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
        model = tv_models.resnet18(weights=None) # Use weights=None for consistency
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1) # Output 1 logit for binary classification
        return model

    class NSFWDataset(torch.utils.data.Dataset):
        def __init__(self, filepaths, labels, transform=None, img_size=224, **kwargs): # Added img_size default
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
                # Create a placeholder image if loading fails
                print(f"Warning: Could not load image {self.filepaths[idx]}. Using placeholder.")
                img = Image.new("RGB", (self.img_size, self.img_size), color="grey")

            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float)

    def load_data_from_module(): # Dummy load_data
        print("WARNING: Using DUMMY load_data. Please ensure your data paths are correct.")
        # This part needs to be adapted to your actual data structure if fallback is used.
        # For example, create dummy file paths and labels
        dummy_files = [str(PROJECT_ROOT / f"dummy_img_{i}.png") for i in range(100)]
        dummy_labels = [i % 2 for i in range(100)]
        # Create dummy image files for the dummy data to work if NSFWDataset tries to open them
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

SHAP_DATA_FILES_LIST_PATH = LEARNING_RESULTS_DIR / "final_test_data_paths.txt"
SHAP_DATA_LABELS_LIST_PATH = LEARNING_RESULTS_DIR / "final_test_data_labels.txt"

N_BACKGROUND_SAMPLES = 100  # Reduced for faster testing if needed, use 50-100 for real runs
N_EXPLAIN_SAMPLES = 20


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


def get_data_for_shap(filepaths_all, labels_all, transform, num_background, num_explain, device):
    if not filepaths_all or not labels_all:
        print("Error: filepaths_all or labels_all is empty.")
        return None, None, None, None

    if len(filepaths_all) < num_background + num_explain:
        print(f"Not enough samples for SHAP analysis. Have {len(filepaths_all)}, need {num_background + num_explain}.")
        # Optionally, adjust num_background and num_explain if possible
        if len(filepaths_all) < num_explain:
            print("Not even enough samples to explain. Aborting SHAP data prep.")
            return None, None, None, None
        if len(filepaths_all) < num_background:
            num_background = max(1, len(filepaths_all) - num_explain) # Ensure at least 1 background if possible
            print(f"Adjusting num_background to {num_background}")

    # Ensure filepaths and labels are lists, not numpy arrays for shuffling with indices
    filepaths_all = list(filepaths_all)
    labels_all = list(labels_all)


    indices = np.random.permutation(len(filepaths_all))
    # Ensure filepaths_all and labels_all are subscriptable by the permuted indices
    try:
        filepaths_shuffled = [filepaths_all[i] for i in indices]
        labels_shuffled = [labels_all[i] for i in indices]
    except IndexError as e:
        print(f"Error during shuffling: {e}. Lengths: filepaths={len(filepaths_all)}, labels={len(labels_all)}, indices_max={np.max(indices) if len(indices)>0 else 'N/A'}")
        return None, None, None, None


    background_fps = filepaths_shuffled[:num_background]
    explain_fps = filepaths_shuffled[num_background : num_background + num_explain]
    background_lbls = labels_shuffled[:num_background]
    explain_lbls = labels_shuffled[num_background : num_background + num_explain]

    if not background_fps or not explain_fps:
        print("Error: background_fps or explain_fps became empty after slicing.")
        return None, None, None, None

    try:
        background_dataset = NSFWDataset(background_fps, background_lbls, transform=transform)
        explain_dataset = NSFWDataset(explain_fps, explain_lbls, transform=transform)

        background_data_list = []
        for i in range(len(background_dataset)):
            img_tensor, _ = background_dataset[i]
            background_data_list.append(img_tensor)
        
        explain_data_list = []
        for i in range(len(explain_dataset)):
            img_tensor, _ = explain_dataset[i]
            explain_data_list.append(img_tensor)

        if not background_data_list or not explain_data_list:
            print("Error: No data loaded into background_data_list or explain_data_list.")
            return None, None, None, None

        background_data = torch.stack(background_data_list).to(device)
        explain_data = torch.stack(explain_data_list).to(device)
    except Exception as e:
        print(f"Error creating datasets or stacking tensors: {e}")
        return None, None, None, None


    return background_data, explain_data, explain_fps, explain_lbls


def main_shap_analysis():
    print("--- SHAP Analysis ---")
    print(f"Using device: {DEVICE}")

    model = load_trained_model(BEST_MODEL_PATH, BEST_OPTUNA_PARAMS_PATH, DEVICE)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    all_filepaths = []
    all_labels = []

    # Попытка загрузить подготовленные данные
    if SHAP_DATA_FILES_LIST_PATH.exists() and SHAP_DATA_LABELS_LIST_PATH.exists():
        print(f"Loading SHAP data from {SHAP_DATA_FILES_LIST_PATH} and {SHAP_DATA_LABELS_LIST_PATH}")
        try:
            with open(SHAP_DATA_FILES_LIST_PATH, "r") as f:
                all_filepaths = [line.strip() for line in f.readlines() if line.strip()]
            with open(SHAP_DATA_LABELS_LIST_PATH, "r") as f:
                all_labels = [int(line.strip()) for line in f.readlines() if line.strip()]
            if len(all_filepaths) != len(all_labels):
                print("Warning: Mismatch in lengths of filepaths and labels loaded from files. This might cause issues.")
            if not all_filepaths:
                print("Warning: Loaded filepaths list is empty.")
        except Exception as e:
            print(f"Error loading data from file lists: {e}")
            all_filepaths, all_labels = [], [] # Reset if error
    else:
        print(f"SHAP test data files not found at {SHAP_DATA_FILES_LIST_PATH} or {SHAP_DATA_LABELS_LIST_PATH}.")

    if not all_filepaths or not all_labels:
        print("Attempting to load sample data using load_data_from_module (from learning script or dummy)...")
        try:
            # Use the imported (potentially dummy) load_data function
            temp_X, temp_y = load_data_from_module()
            all_filepaths, all_labels = temp_X, temp_y
            print(f"Loaded {len(all_filepaths)} filepaths and {len(all_labels)} labels from load_data_from_module.")
        except Exception as e:
            print(f"Could not load sample data using load_data_from_module: {e}")
            print("SHAP analysis cannot proceed without data.")
            return

    if not all_filepaths or not all_labels:
        print("No data available for SHAP analysis. Exiting.")
        return

    background_data, explain_data, explain_fps, explain_lbls = get_data_for_shap(
        all_filepaths, all_labels, val_transform, N_BACKGROUND_SAMPLES, N_EXPLAIN_SAMPLES, DEVICE
    )

    if background_data is None or explain_data is None:
        print("Failed to prepare SHAP data. Exiting.")
        return

    print(f"Background data shape: {background_data.shape}")
    print(f"Explain data shape: {explain_data.shape}")

    print("Creating GradientExplainer...")
    try:
        # Ensure model is callable and returns a single output (logit) per sample for GradientExplainer
        # If your model outputs probabilities (e.g. after sigmoid), GradientExplainer might work,
        # but typically it expects logits.
        explainer = shap.GradientExplainer(model, background_data)
    except Exception as e:
        print(f"Error creating GradientExplainer: {e}")
        return

    print(f"Calculating SHAP values for {explain_data.shape[0]} samples...")
    try:
        # For binary classification with a single output logit, shap_values will have
        # shape (N_explain_samples, C, H, W, 1) or (N_explain_samples, C, H, W)
        # depending on the exact SHAP version and model output.
        shap_values = explainer.shap_values(explain_data)
        print(f"SHAP values calculated. Shape: {np.array(shap_values).shape}")
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        import traceback
        traceback.print_exc()
        return

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print("Plotting SHAP explanations...")
    for i in range(explain_data.shape[0]):
        original_img_tensor = explain_data[i].cpu() # Shape (C, H, W)
        # Denormalize and prepare original image for plotting
        img_for_plot = original_img_tensor.numpy().transpose(1, 2, 0) # Shape (H, W, C)
        img_for_plot = std * img_for_plot + mean # Denormalize
        img_for_plot = np.clip(img_for_plot, 0, 1)

        plt.figure(figsize=(12, 6)) # Adjusted figsize slightly
        plt.subplot(1, 2, 1)
        plt.imshow(img_for_plot) # Expects (H, W, C)
        true_label_str = 'NSFW' if explain_lbls[i] == 1 else 'Regular'
        plt.title(f"Original Image {i+1}\nFile: ...{Path(explain_fps[i]).name}\nTrue Label: {true_label_str}", fontsize=10)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        try:
            # shap_values is expected to be (N_samples, C, H, W, num_outputs_classes)
            # For binary with 1 logit, num_outputs_classes is 1.
            # So, shap_values[i] (current_shap_output_class) has shape (C, H, W, 1) or (C, H, W)
            current_shap_output_class = shap_values[i] # This gets SHAP values for the first (and only) output class

            # The output from GradientExplainer for a model with a single output neuron
            # might be (C, H, W) directly or (C, H, W, 1).
            # If it's (C,H,W,1), we need to squeeze the last dimension.
            if current_shap_output_class.ndim == 4 and current_shap_output_class.shape[-1] == 1:
                shap_map_chw = current_shap_output_class.squeeze(-1) # Shape: (C, H, W)
            elif current_shap_output_class.ndim == 3: # Should be (C, H, W)
                shap_map_chw = current_shap_output_class
            else:
                raise ValueError(f"SHAP values for sample {i} have unexpected shape {current_shap_output_class.shape}. Expected (C,H,W) or (C,H,W,1).")

            # To visualize, we need to aggregate SHAP values across channels.
            # Summing absolute values is a common way: (H, W)
            shap_heatmap_hw = np.abs(shap_map_chw).sum(axis=0)

            plt.imshow(shap_heatmap_hw, cmap='viridis') # cmap='coolwarm' is also good for SHAP
            plt.colorbar(label="Sum of abs(SHAP values) across channels")
            plt.title(f"SHAP Explanation {i+1}", fontsize=10)
            plt.axis("off")

        except Exception as e:
            print(f"Error plotting SHAP for sample {i}: {e}")
            import traceback
            print(traceback.format_exc())
            plt.text(0.5, 0.5, "Error in SHAP plot", ha="center", va="center", transform=plt.gca().transAxes)
            plt.axis("off")

        plt.tight_layout()
        save_path = SHAP_RESULTS_DIR / f"shap_explanation_{Path(explain_fps[i]).stem}.png"
        try:
            plt.savefig(save_path)
            print(f"Saved explanation: {save_path}")
        except Exception as e_save:
            print(f"Error saving plot {save_path}: {e_save}")
        plt.close() # Close the figure to free memory

    print("--- SHAP Analysis Finished ---")


if __name__ == "__main__":
    # Set a fixed CWD for reproducibility if paths are relative in module
    # Or better, ensure all paths in the module are absolute or resolvable from PROJECT_ROOT
    # For now, assuming paths are handled correctly.
    main_shap_analysis()