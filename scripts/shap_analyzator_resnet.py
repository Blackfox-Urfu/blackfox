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

# Make sure Path is imported

# Get the absolute path to the project root (blackfox)
# Assuming this script is in PROJECT_ROOT/scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Attempt to import from the main learning script ---
LEARNING_SCRIPT_MODULE_PATH = "app.learn.resnet_learn_slut_detector"

try:
    module = __import__(LEARNING_SCRIPT_MODULE_PATH,
                        fromlist=['DEVICE', 'MODEL_DIR', 'val_transform', 'BEST_MODEL_PATH',
                                  'BEST_OPTUNA_PARAMS_PATH', 'RESULTS_DIR', 'NSFWDataset',
                                  'create_configurable_model'])

    DEVICE = module.DEVICE
    MODEL_DIR_str = module.MODEL_DIR  # Store original
    val_transform = module.val_transform
    BEST_MODEL_PATH_str = module.BEST_MODEL_PATH
    BEST_OPTUNA_PARAMS_PATH_str = module.BEST_OPTUNA_PARAMS_PATH
    LEARNING_RESULTS_DIR_str = module.RESULTS_DIR
    NSFWDataset = module.NSFWDataset
    create_configurable_model = module.create_configurable_model

    print(f"Successfully imported components from {LEARNING_SCRIPT_MODULE_PATH}")

    # Convert all paths to absolute using PROJECT_ROOT
    MODEL_DIR = Path(MODEL_DIR_str).resolve()
    if not MODEL_DIR.is_absolute():
        MODEL_DIR = (PROJECT_ROOT / MODEL_DIR_str).resolve()

    BEST_MODEL_PATH = Path(BEST_MODEL_PATH_str).resolve()
    if not BEST_MODEL_PATH.is_absolute():
        BEST_MODEL_PATH = (MODEL_DIR / Path(BEST_MODEL_PATH_str).name).resolve()

    BEST_OPTUNA_PARAMS_PATH = Path(BEST_OPTUNA_PARAMS_PATH_str).resolve()
    if not BEST_OPTUNA_PARAMS_PATH.is_absolute():
        BEST_OPTUNA_PARAMS_PATH = (MODEL_DIR / BEST_OPTUNA_PARAMS_PATH_str).resolve()

    LEARNING_RESULTS_DIR = Path(LEARNING_RESULTS_DIR_str).resolve()
    if not LEARNING_RESULTS_DIR.is_absolute():
        LEARNING_RESULTS_DIR = (MODEL_DIR / LEARNING_RESULTS_DIR_str).resolve()

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
        print("WARNING: Using DUMMY create_configurable_model. Real one must be imported for correct analysis.")
        from torchvision import models as tv_models
        model = tv_models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        return model

    class NSFWDataset(torch.utils.data.Dataset):
        def __init__(self, filepaths, labels, transform=None, **kwargs):
            self.filepaths = filepaths
            self.labels = labels
            self.transform = transform
            self.img_size = kwargs.get('img_size', 224)

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.filepaths[idx]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (self.img_size, self.img_size), color="grey")
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float)

# --- Configuration for SHAP analysis ---
SHAP_RESULTS_DIR = (PROJECT_ROOT / "shap_analysis_results").resolve()
os.makedirs(SHAP_RESULTS_DIR, exist_ok=True)

SHAP_DATA_FILES_LIST_PATH = (LEARNING_RESULTS_DIR / "final_test_data_paths.txt").resolve()
SHAP_DATA_LABELS_LIST_PATH = (LEARNING_RESULTS_DIR / "final_test_data_labels.txt").resolve()

N_BACKGROUND_SAMPLES = 50
N_EXPLAIN_SAMPLES = 10


def load_trained_model(model_path, params_path, device):
    model_path_str = str(model_path)
    params_path_str = str(params_path)

    if not os.path.exists(model_path_str):
        print(f"Model file not found: {model_path_str}")
        return None
    if not os.path.exists(params_path_str):
        print(f"Model parameters file not found: {params_path_str}")
        return None

    try:
        best_params = joblib.load(params_path_str)
        print(f"Loaded model parameters from {params_path_str}")
    except Exception as e:
        print(f"Error loading parameters from {params_path_str}: {e}")
        return None

    try:
        model = create_configurable_model(best_params)
        model.load_state_dict(torch.load(model_path_str, map_location=device))
        model.eval()
        model.to(device)
        print(f"Model loaded from {model_path_str} and set to eval mode on {device}.")
        return model
    except Exception as e:
        print(f"Error creating or loading model state_dict: {e}")
        print("This often means the model architecture defined by 'create_configurable_model' (and params) "
              "does not match the architecture of the saved model state_dict.")
        return None


def get_data_for_shap(filepaths_all, labels_all, transform, num_background, num_explain, device):
    if len(filepaths_all) < (num_background + num_explain):
        print(
            f"Warning: Not enough data samples ({len(filepaths_all)}) for requested background ({num_background}) and explain ({num_explain}) samples."
        )
        if len(filepaths_all) < 10:
            return None, None, None, None
        num_explain = max(1, len(filepaths_all) // 10)
        num_background = max(5, len(filepaths_all) - num_explain)

    indices = np.arange(len(filepaths_all))
    np.random.shuffle(indices)
    shuffled_filepaths = [filepaths_all[i] for i in indices]
    shuffled_labels = [labels_all[i] for i in indices]

    background_filepaths = shuffled_filepaths[:num_background]
    background_labels = shuffled_labels[:num_background]
    explain_filepaths = shuffled_filepaths[num_background: num_background + num_explain]
    explain_labels = shuffled_labels[num_background: num_background + num_explain]

    print(f"Preparing {len(background_filepaths)} background samples and {len(explain_filepaths)} explain samples.")

    background_dataset = NSFWDataset(background_filepaths, background_labels, transform=transform, cache_ram=False)
    explain_dataset = NSFWDataset(explain_filepaths, explain_labels, transform=transform, cache_ram=False)

    background_data_list = [background_dataset[i][0] for i in tqdm(range(len(background_dataset)), desc="Loading background data")]
    explain_data_list = [explain_dataset[i][0] for i in tqdm(range(len(explain_dataset)), desc="Loading explain data")]

    if not background_data_list or not explain_data_list:
        print("Error: Failed to load background or explain data.")
        return None, None, None, None

    background_tensor = torch.stack(background_data_list).to(device)
    explain_tensor = torch.stack(explain_data_list).to(device)

    return background_tensor, explain_tensor, explain_filepaths, explain_labels


def main_shap_analysis():
    print(f"--- SHAP Analysis ---")
    print(f"Using device: {DEVICE}")

    model = load_trained_model(BEST_MODEL_PATH, BEST_OPTUNA_PARAMS_PATH, DEVICE)
    if model is None:
        print("Failed to load model. Exiting SHAP analysis.")
        return

    all_filepaths_for_shap = []
    all_labels_for_shap = []

    shap_data_files_list_path_str = str(SHAP_DATA_FILES_LIST_PATH)
    shap_data_labels_list_path_str = str(SHAP_DATA_LABELS_LIST_PATH)

    if os.path.exists(shap_data_files_list_path_str) and os.path.exists(shap_data_labels_list_path_str):
        print(f"Loading SHAP data from {shap_data_files_list_path_str} and {shap_data_labels_list_path_str}")
        with open(shap_data_files_list_path_str, "r") as f:
            all_filepaths_for_shap = [line.strip() for line in f.readlines()]
        with open(shap_data_labels_list_path_str, "r") as f:
            all_labels_for_shap = [int(line.strip()) for line in f.readlines()]

    if not all_filepaths_for_shap or not all_labels_for_shap:
        print(f"SHAP data files not found or empty: {shap_data_files_list_path_str} or {shap_data_labels_list_path_str}")
        print("Attempting to load sample data directly for SHAP demonstration (not recommended for final analysis)...")

        try:
            main_data_module = __import__(LEARNING_SCRIPT_MODULE_PATH, fromlist=['load_data'])
            load_main_data_func = main_data_module.load_data
            temp_X, temp_y = load_main_data_func()

            if len(temp_X) > N_BACKGROUND_SAMPLES + N_EXPLAIN_SAMPLES:
                from sklearn.model_selection import train_test_split as shap_tts
                sample_size = N_BACKGROUND_SAMPLES + N_EXPLAIN_SAMPLES + 10
                current_data_size = len(temp_X)
                test_fraction = min(0.1, sample_size / current_data_size if current_data_size > 0 else 0.1)

                if current_data_size <= sample_size:
                    all_filepaths_for_shap, all_labels_for_shap = temp_X, temp_y
                else:
                    _, all_filepaths_for_shap, _, all_labels_for_shap = shap_tts(
                        temp_X, temp_y, test_size=test_fraction, stratify=temp_y, random_state=123
                    )

                print(f"Loaded {len(all_filepaths_for_shap)} sample images for SHAP.")
            else:
                print(f"Not enough data ({len(temp_X)}) to load directly for SHAP demonstration. Exiting.")
                return
        except (ImportError, AttributeError) as e_load:
            print(f"Could not load sample data due to: {e_load}. Exiting SHAP analysis.")
            return
        except Exception as e_data:
            print(f"An unexpected error occurred while loading sample data: {e_data}. Exiting SHAP analysis.")
            return

    background_data, explain_data, explain_fps, explain_lbls = get_data_for_shap(
        all_filepaths_for_shap, all_labels_for_shap, val_transform,
        N_BACKGROUND_SAMPLES, N_EXPLAIN_SAMPLES, DEVICE
    )

    if background_data is None or explain_data is None:
        print("Failed to prepare data for SHAP. Exiting.")
        return

    print("Creating SHAP DeepExplainer...")
    explainer = shap.DeepExplainer(model, background_data)

    print(f"Calculating SHAP values for {explain_data.shape[0]} samples...")
    try:
        shap_values = explainer.shap_values(explain_data)
    except Exception as e:
        print(f"Error calculating SHAP values with DeepExplainer: {e}")
        print("Consider checking model compatibility or trying GradientExplainer.")
        return

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print("Plotting SHAP explanations...")
    for i in range(explain_data.shape[0]):
        original_img_tensor = explain_data[i].cpu()
        img_for_plot = original_img_tensor.numpy().transpose(1, 2, 0)
        img_for_plot = std * img_for_plot + mean
        img_for_plot = np.clip(img_for_plot, 0, 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_for_plot)
        true_label_str = 'NSFW' if explain_lbls[i] == 1 else 'Regular'
        plt.title(f"Original Image {i+1}\nFile: ...{Path(explain_fps[i]).name}\nTrue Label: {true_label_str}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        try:
            pixels_for_shap_plot = explain_data[i].cpu().numpy().transpose(1, 2, 0)
            shap_values_single_transposed = shap_values[i].transpose(1, 2, 0)
            shap.image_plot(shap_values_single_transposed, pixels_for_shap_plot, show=False)
            plt.title(f"SHAP Explanation {i+1}")
        except Exception as e_plot:
            print(f"Error during shap.image_plot for sample {i}: {e_plot}")
            plt.text(0.5, 0.5, "Error in SHAP plot", ha="center", va="center")

        plt.tight_layout()
        save_path = SHAP_RESULTS_DIR / f"shap_explanation_{Path(explain_fps[i]).stem}.png"
        plt.savefig(str(save_path))
        plt.close()
        print(f"Saved SHAP explanation for sample {i+1} to {save_path}")

    print("--- SHAP Analysis Finished ---")


if __name__ == "__main__":
    main_shap_analysis()