import shap
import os 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
import matplotlib.pyplot as plt

# Установка количества ядер для параллельных вычислений
shap.initjs()  # Инициализация JS для интерактивных визуализаций

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)

    ad_texts = [extract_message_data(msg) for msg in ad_data['messages'] if clean_text(extract_text(msg))]
    non_ad_texts = [extract_message_data(msg) for msg in non_ad_data['messages'] if clean_text(extract_text(msg))]

    texts = [clean_text(msg['text']) for msg in ad_texts + non_ad_texts]
    labels = [1] * len(ad_texts) + [0] * len(non_ad_texts)

    return texts, labels

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', '')

def extract_text(message):
    full_text = ""
    for part in message["text"]:
        if isinstance(part, dict):
            full_text += part["text"]
        else:
            full_text += part
    return full_text

def extract_message_data(message):
    return {
        "text": extract_text(message),
        "date": message.get("date", ""),
        "from": message.get("from", ""),
        "photo": message.get("photo", ""),
        "file_name": message.get("file_name", "")
    }

# Основной код
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ad_filepath = os.path.join(BASE_DIR, 'data/reklama', 'result.json')
non_ad_filepath = os.path.join(BASE_DIR, 'data/nereklama', 'result.json')
texts, labels = process_data(ad_filepath, non_ad_filepath)

# Загрузка модели и векторизатора
model = joblib.load('randfor_model.pkl')
vectorizer = joblib.load('randfor_vectorizer.pkl')

# Векторизация и подготовка данных
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
test_vectors = vectorizer.transform(test_texts)
feature_names = vectorizer.get_feature_names_out()
X_test = pd.DataFrame(test_vectors.toarray(), columns=feature_names)

# Инициализация explainer
explainer = shap.TreeExplainer(model)

# Вычисление SHAP значений
shap_values = explainer.shap_values(X_test, check_additivity=False)

# Функция для сохранения визуализаций в SVG
def save_shap_plot(plot_func, filename, *args, **kwargs):
    plt.figure()
    plot_func(*args, **kwargs)
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()

# Сохранение всех визуализаций
output_dir = "shap_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Summary plot
save_shap_plot(shap.summary_plot, 
              f"{output_dir}/summary_plot.svg",
              shap_values[1], X_test, show=False)

# 2. Force plot (сохраняем первые 5 примеров)
for i in range(5):
    force_plot = shap.force_plot(explainer.expected_value[1],
                               shap_values[1][i,:],
                               X_test.iloc[i,:],
                               matplotlib=True,
                               show=False)
    save_shap_plot(lambda: force_plot, 
                  f"{output_dir}/force_plot_{i}.svg")

# 3. Waterfall plots (первые 5 примеров)
for i in range(5):
    save_shap_plot(shap.plots.waterfall,
                  f"{output_dir}/waterfall_{i}.svg",
                  shap.Explanation(values=shap_values[1][i],
                                 base_values=explainer.expected_value[1],
                                 data=X_test.iloc[i]))  

# 4. Decision plot 
save_shap_plot(shap.decision_plot,
              f"{output_dir}/decision_plot.svg",
              explainer.expected_value[1],
              shap_values[1][:100],  # Первые 100 примеров
              feature_names=feature_names,
              show=False)

print(f"Все визуализации сохранены в формате SVG в директории {output_dir}")