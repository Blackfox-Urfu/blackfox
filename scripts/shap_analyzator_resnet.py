import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from tqdm import tqdm

# Предполагается, что скрипт обучения (resnet_learn_slut_detector.py)
# находится в той же директории или доступен для импорта.
# Если он в другом месте, нужно настроить sys.path или сделать его пакетом.
try:
    # Попытка импортировать необходимые компоненты из скрипта обучения
    from resnet_learn_slut_detector import (
        create_configurable_model,
        NSFWDataset,
        val_transform, # Используем val_transform для анализа
        DEVICE, # Используем то же устройство
        MODEL_DIR, # Путь к директории с моделью
        BEST_MODEL_PATH,
        BEST_OPTUNA_PARAMS_PATH,
        RESULTS_DIR as LEARNING_RESULTS_DIR # Чтобы не путать с результатами SHAP
    )
except ImportError as e:
    print(f"Error importing from resnet_learn_slut_detector.py: {e}")
    print("Make sure resnet_learn_slut_detector.py is in the same directory or PYTHONPATH.")
    # Зададим дефолтные значения, если импорт не удался, чтобы скрипт хотя бы загрузился
    # Но для реальной работы импорт должен быть успешным.
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_DIR = 'model/optuna_resnet' # Должен совпадать с тем, что в скрипте обучения
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_optuna_resnet.pth')
    BEST_OPTUNA_PARAMS_PATH = os.path.join(MODEL_DIR, 'best_optuna_params.pkl')
    LEARNING_RESULTS_DIR = os.path.join(MODEL_DIR, 'results')
    # val_transform и create_configurable_model нужно будет определить здесь, если импорт не работает

    # Заглушка для val_transform, если импорт не удался
    from torchvision import transforms as T
    IMG_SIZE_SHAP = 224 # Должно совпадать с IMG_SIZE в обучении
    val_transform = T.Compose([
        T.Resize((IMG_SIZE_SHAP, IMG_SIZE_SHAP)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Заглушка для create_configurable_model
    def create_configurable_model(params):
        print("WARNING: Using dummy create_configurable_model. Real one should be imported.")
        from torchvision import models as tv_models
        model = tv_models.resnet18(weights=None) # Простая заглушка
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        return model
    # Заглушка для NSFWDataset
    class NSFWDataset(torch.utils.data.Dataset):
        def __init__(self, filepaths, labels, transform=None, **kwargs):
            self.filepaths = filepaths; self.labels = labels; self.transform = transform
        def __len__(self): return len(self.filepaths)
        def __getitem__(self, idx):
            try: img = Image.open(self.filepaths[idx]).convert('RGB')
            except: img = Image.new('RGB', (IMG_SIZE_SHAP, IMG_SIZE_SHAP), color='grey')
            if self.transform: img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float)


# --- Конфигурация для SHAP анализа ---
SHAP_RESULTS_DIR = 'shap_analysis_results'
os.makedirs(SHAP_RESULTS_DIR, exist_ok=True)

# Пути к данным для SHAP анализа (например, часть тестового набора из основного скрипта)
# Вам нужно будет предоставить эти пути.
# Можно взять из X_test_final, y_test_final, которые использовались в main() скрипта обучения.
# Для примера, предположим, что у вас есть файлы со списком путей:
SHAP_DATA_FILES_LIST_PATH = os.path.join(LEARNING_RESULTS_DIR, "final_test_data_paths.txt") # Пример
SHAP_DATA_LABELS_LIST_PATH = os.path.join(LEARNING_RESULTS_DIR, "final_test_data_labels.txt") # Пример

# Количество сэмплов для фона (background) и для объяснения (explain)
N_BACKGROUND_SAMPLES = 50  # Меньше для скорости, больше для точности
N_EXPLAIN_SAMPLES = 10     # Количество изображений, для которых будем строить детальные объяснения


def load_trained_model(model_path, params_path, device):
    """Загружает обученную модель и ее параметры."""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    if not os.path.exists(params_path):
        print(f"Model parameters file not found: {params_path}")
        return None

    try:
        best_params = joblib.load(params_path)
        print(f"Loaded model parameters from {params_path}")
    except Exception as e:
        print(f"Error loading parameters from {params_path}: {e}")
        return None

    try:
        model = create_configurable_model(best_params) # Используем импортированную функцию
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        print(f"Model loaded from {model_path} and set to eval mode on {device}.")
        return model
    except Exception as e:
        print(f"Error creating or loading model state_dict: {e}")
        return None

def get_data_for_shap(filepaths_all, labels_all, transform, num_background, num_explain, device):
    """
    Подготавливает фоновые данные и данные для объяснения.
    Возвращает тензоры данных и, опционально, пути к файлам для объяснения.
    """
    if len(filepaths_all) < (num_background + num_explain):
        print(f"Warning: Not enough data samples ({len(filepaths_all)}) for requested background ({num_background}) and explain ({num_explain}) samples.")
        # Уменьшаем количество, если данных не хватает, или возвращаем None
        if len(filepaths_all) < 10 : return None, None, None, None # Слишком мало для анализа
        num_explain = max(1, len(filepaths_all) // 10)
        num_background = max(5, len(filepaths_all) - num_explain)


    # Перемешиваем данные один раз, чтобы выборки были случайными
    indices = np.arange(len(filepaths_all))
    np.random.shuffle(indices)
    
    shuffled_filepaths = [filepaths_all[i] for i in indices]
    shuffled_labels = [labels_all[i] for i in indices]

    # Выбираем данные для фона
    background_filepaths = shuffled_filepaths[:num_background]
    background_labels = shuffled_labels[:num_background]
    
    # Выбираем данные для объяснения
    explain_filepaths = shuffled_filepaths[num_background : num_background + num_explain]
    explain_labels = shuffled_labels[num_background : num_background + num_explain]

    print(f"Preparing {len(background_filepaths)} background samples and {len(explain_filepaths)} explain samples.")

    background_dataset = NSFWDataset(background_filepaths, background_labels, transform=transform, cache_ram=False)
    explain_dataset = NSFWDataset(explain_filepaths, explain_labels, transform=transform, cache_ram=False)

    # Не используем DataLoader, чтобы получить тензоры напрямую, если это удобно для SHAP
    # Или можно использовать DataLoader с batch_size=1
    
    background_data_list = []
    for i in tqdm(range(len(background_dataset)), desc="Loading background data"):
        img, _ = background_dataset[i]
        background_data_list.append(img)
    
    explain_data_list = []
    for i in tqdm(range(len(explain_dataset)), desc="Loading explain data"):
        img, _ = explain_dataset[i]
        explain_data_list.append(img)

    if not background_data_list or not explain_data_list:
        print("Error: Failed to load background or explain data.")
        return None, None, None, None

    background_tensor = torch.stack(background_data_list).to(device)
    explain_tensor = torch.stack(explain_data_list).to(device)
    
    return background_tensor, explain_tensor, explain_filepaths, explain_labels


def main_shap_analysis():
    print(f"--- SHAP Analysis ---")
    print(f"Using device: {DEVICE}")

    # 1. Загрузка модели
    model = load_trained_model(BEST_MODEL_PATH, BEST_OPTUNA_PARAMS_PATH, DEVICE)
    if model is None:
        print("Failed to load model. Exiting SHAP analysis.")
        return

    # 2. Загрузка данных для SHAP
    # Вам нужно будет создать эти файлы `final_test_data_paths.txt` и `final_test_data_labels.txt`
    # во время финальной оценки в основном скрипте обучения.
    # Пример, как их можно было бы создать в resnet_learn_slut_detector.py ПОСЛЕ train_test_split:
    #
    # if not os.path.exists(SHAP_DATA_FILES_LIST_PATH) and X_test_final:
    #     with open(SHAP_DATA_FILES_LIST_PATH, 'w') as f:
    #         for item in X_test_final:
    #             f.write(f"{item}\n")
    #     with open(SHAP_DATA_LABELS_LIST_PATH, 'w') as f:
    #         for item in y_test_final:
    #             f.write(f"{item}\n")
    #     print(f"Saved file paths and labels for SHAP analysis to {LEARNING_RESULTS_DIR}")

    all_filepaths_for_shap = []
    all_labels_for_shap = []

    if os.path.exists(SHAP_DATA_FILES_LIST_PATH) and os.path.exists(SHAP_DATA_LABELS_LIST_PATH):
        print(f"Loading SHAP data from {SHAP_DATA_FILES_LIST_PATH} and {SHAP_DATA_LABELS_LIST_PATH}")
        with open(SHAP_DATA_FILES_LIST_PATH, 'r') as f:
            all_filepaths_for_shap = [line.strip() for line in f.readlines()]
        with open(SHAP_DATA_LABELS_LIST_PATH, 'r') as f:
            all_labels_for_shap = [int(line.strip()) for line in f.readlines()]
        if not all_filepaths_for_shap or not all_labels_for_shap:
            print("SHAP data files are empty.")
            return
    else:
        print(f"SHAP data files not found: {SHAP_DATA_FILES_LIST_PATH} or {SHAP_DATA_LABELS_LIST_PATH}")
        print("Please generate these files from your main training script's test set.")
        # Для демонстрации можно использовать небольшое количество случайных данных, если пути не найдены
        # Но для реального анализа нужны релевантные данные.
        # Здесь можно добавить загрузку из SLUT_DATA_DIR и REGULAR_DATA_DIR, если файлы не найдены
        print("Attempting to load sample data directly for SHAP demonstration (not recommended for final analysis)...")
        from resnet_learn_slut_detector import load_data as load_main_data
        temp_X, temp_y = load_main_data()
        if len(temp_X) > N_BACKGROUND_SAMPLES + N_EXPLAIN_SAMPLES:
            from sklearn.model_selection import train_test_split as shap_tts
            _, all_filepaths_for_shap, _, all_labels_for_shap = shap_tts(temp_X, temp_y, test_size=0.05, stratify=temp_y, random_state=123) # Берем 5% для примера
            if len(all_filepaths_for_shap) < N_BACKGROUND_SAMPLES + N_EXPLAIN_SAMPLES:
                 print("Not enough sample data loaded directly. Exiting SHAP analysis.")
                 return
            print(f"Loaded {len(all_filepaths_for_shap)} sample images for SHAP.")
        else:
            print("Not enough data to load directly for SHAP demonstration. Exiting.")
            return


    background_data, explain_data, explain_fps, explain_lbls = get_data_for_shap(
        all_filepaths_for_shap, all_labels_for_shap, val_transform,
        N_BACKGROUND_SAMPLES, N_EXPLAIN_SAMPLES, DEVICE
    )

    if background_data is None or explain_data is None:
        print("Failed to prepare data for SHAP. Exiting.")
        return

    # 3. Создание SHAP Explainer
    # SHAP ожидает, что модель возвращает выход для каждого класса.
    # Наша модель возвращает один логит (до сигмоиды).
    # Для DeepExplainer лучше подавать на вход тензоры.
    # Для KernelExplainer можно использовать функцию-обертку.
    # DeepExplainer обычно предпочтительнее для глубоких сетей.

    print("Creating SHAP DeepExplainer...")
    # DeepExplainer ожидает модель и фоновые данные.
    # Модель должна быть на том же устройстве, что и данные.
    explainer = shap.DeepExplainer(model, background_data)

    # 4. Вычисление SHAP values
    print(f"Calculating SHAP values for {explain_data.shape[0]} samples...")
    try:
        shap_values = explainer.shap_values(explain_data)
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        print("This might be due to model architecture or data shapes. Try a smaller background/explain set or KernelExplainer.")
        
        # Попытка с KernelExplainer как запасной вариант
        print("\nAttempting with KernelExplainer (slower)...")
        
        def predict_logits(data_np_array):
            # data_np_array будет (N, C, H, W) или (N, H, W, C) в зависимости от SHAP
            # Нужно привести к (N, C, H, W) и в тензор torch
            if data_np_array.ndim == 4 and data_np_array.shape[-1] == 3 : # (N, H, W, C)
                data_np_array = data_np_array.transpose(0, 3, 1, 2)
            
            data_tensor = torch.tensor(data_np_array, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = model(data_tensor)
            return logits.cpu().numpy() # Возвращаем логиты (N, 1)

        # Для KernelExplainer фоновые данные могут быть представлены медианой или k-средними
        # background_summary = shap.kmeans(background_data.cpu().numpy(), 10) # 10 центроидов
        # Или просто передать тензор, KernelExplainer сам разберется
        
        # KernelExplainer может быть очень медленным для изображений.
        # Для него лучше преобразовать данные в нужный формат (обычно (N, H, W, C) или (N, C, H, W))
        # и убедиться, что predict_logits это понимает.
        # Для изображений часто используют PartitionExplainer или GradientExplainer.
        # Но DeepExplainer - хороший старт, если работает.
        # Если DeepExplainer не работает, и KernelExplainer слишком медленный,
        # нужно будет глубже разбираться с совместимостью модели и SHAP explainer'ов для изображений.
        # Это может потребовать маскировки входов и т.д.

        # Простой KernelExplainer (может быть очень медленным!)
        # kernel_explainer = shap.KernelExplainer(predict_logits, background_data.cpu().numpy()[:10]) # Меньше фона для скорости
        # try:
        #     shap_values_kernel = kernel_explainer.shap_values(explain_data.cpu().numpy(), nsamples=50) # nsamples - для аппроксимации
        #     shap_values = shap_values_kernel # Если удалось
        # except Exception as ke:
        #     print(f"KernelExplainer also failed: {ke}")
        #     return
        print("KernelExplainer attempt skipped for brevity, focus on making DeepExplainer work or use GradientExplainer.")
        return


    # shap_values для бинарной классификации с одним выходом будет списком с одним элементом,
    # содержащим массив (N_EXPLAIN_SAMPLES, C, H, W)
    if isinstance(shap_values, list):
        shap_values_for_plot = shap_values[0] # Берем значения для единственного выхода
    else:
        shap_values_for_plot = shap_values

    # Для визуализации нам нужны исходные изображения в формате, понятном matplotlib (H, W, C)
    # и без нормализации.
    # explain_data сейчас (N, C, H, W) и нормализован.
    # Нужно денормализовать и транспонировать.

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 5. Визуализация SHAP values
    print("Plotting SHAP explanations...")
    for i in range(explain_data.shape[0]):
        # Денормализация и изменение порядка осей для отображения изображения
        original_img_tensor = explain_data[i].cpu()
        img_for_plot = original_img_tensor.numpy().transpose(1, 2, 0)
        img_for_plot = std * img_for_plot + mean # Денормализация
        img_for_plot = np.clip(img_for_plot, 0, 1)

        # SHAP values также нужно транспонировать для image_plot
        # shap_values_for_plot имеет форму (N, C, H, W)
        shap_v = shap_values_for_plot[i].transpose(1, 2, 0)

        plt.figure(figsize=(12, 5))
        # Оригинальное изображение
        plt.subplot(1, 2, 1)
        plt.imshow(img_for_plot)
        plt.title(f"Original Image {i+1}\nFile: ...{explain_fps[i][-30:]}\nTrue Label: {'NSFW' if explain_lbls[i]==1 else 'Regular'}")
        plt.axis('off')

        # SHAP Image Plot
        plt.subplot(1, 2, 2)
        # shap.image_plot ожидает shap_values в виде (H,W,C) или (H,W) и пиксели (H,W,C)
        # Если shap_v уже (H,W,C) и img_for_plot (H,W,C)
        # Иногда для image_plot лучше передавать explain_data[i].cpu().numpy() напрямую,
        # если он ожидает (C,H,W) и сам транспонирует.
        # Проверим размерность shap_v
        try:
            # Попробуем передать данные как есть, если SHAP это поддерживает.
            # Для DeepExplainer shap_values для ConvNet обычно (N, C, H, W)
            # Для image_plot ему нужны пиксели (N, H, W, C) или (N, H, W)
            # и shap_values (N, H, W, C) или (N, H, W)
            # Мы передаем единичные сэмплы.
            # shap_values_for_plot[i] это (C,H,W)
            # explain_data[i].cpu().numpy() это (C,H,W)
            
            # shap.image_plot ожидает пиксели в формате (H, W, C) или (H, W)
            # и shap_values в таком же формате или (H, W, C_shap)
            # Наши shap_values (C, H, W), пиксели (C, H, W)
            # Транспонируем для image_plot
            pixels_for_shap_plot = explain_data[i].cpu().numpy().transpose(1,2,0)
            shap_values_single_transposed = shap_values_for_plot[i].transpose(1,2,0)
            
            shap.image_plot(shap_values_single_transposed, 
                            pixels_for_shap_plot, 
                            show=False)
            plt.title(f"SHAP Explanation {i+1}")
        except Exception as e_plot:
            print(f"Error during shap.image_plot for sample {i}: {e_plot}")
            plt.text(0.5, 0.5, "Error in SHAP plot", ha='center', va='center')


        plt.tight_layout()
        save_path = os.path.join(SHAP_RESULTS_DIR, f'shap_explanation_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved SHAP explanation for sample {i+1} to {save_path}")

    # Общий summary plot
    # Для summary_plot нужны shap_values (N, num_features)
    # Для изображений это сложнее. DeepExplainer для изображений возвращает (N, C, H, W).
    # Мы можем сделать summary_plot по средним значениям SHAP для каждого канала, например.
    # Или, если мы хотим видеть важность "суперпикселей", нужно использовать другие методы.
    # Для простого summary_plot можно усреднить по пространственным измерениям и каналам.
    # Но это не очень информативно для изображений.

    # Вместо summary_plot, для изображений часто используют force_plot для отдельных предсказаний
    # или просто image_plot, как выше.

    # Попробуем force_plot для первого объясняемого изображения.
    # Force plot требует ожидаемое значение (base value) из explainer.
    # И shap_values для одного экземпляра.
    if hasattr(explainer, 'expected_value'):
        expected_value = explainer.expected_value
        if isinstance(expected_value, list): # Если выходов несколько (даже если один)
            expected_value = expected_value[0]

        # shap_values_for_plot[0] это (C,H,W) - нам нужно его как-то агрегировать или использовать 
        # версию SHAP, которая работает с многомерными выходами для force_plot.
        # Обычно force_plot используется для табличных данных.
        # Для изображений он менее интуитивен, если не свести признаки к одномерному вектору.
        
        # Пропустим force_plot для изображений в этой базовой реализации,
        # так как он требует дополнительной обработки признаков.
        print("Skipping SHAP force_plot as it's less straightforward for image data with DeepExplainer.")

    else:
        print("Explainer does not have expected_value, cannot create force_plot.")


    print("--- SHAP Analysis Finished ---")


if __name__ == "__main__":
    # Перед запуском убедитесь, что:
    # 1. Скрипт resnet_learn_slut_detector.py находится там, откуда его можно импортировать.
    # 2. Модель (BEST_MODEL_PATH) и ее параметры (BEST_OPTUNA_PARAMS_PATH) существуют.
    # 3. (Опционально, но рекомендуется) Файлы SHAP_DATA_FILES_LIST_PATH и SHAP_DATA_LABELS_LIST_PATH
    #    созданы основным скриптом и содержат пути к данным тестового набора.
    #    Если их нет, скрипт попытается загрузить демонстрационные данные.
    main_shap_analysis()