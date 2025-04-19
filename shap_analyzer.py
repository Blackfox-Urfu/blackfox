import shap
import os 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
import matplotlib.pyplot as plt

# Initialize JS for visualizations
shap.initjs()

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', '').strip()

def extract_text(message):
    full_text = ""
    for part in message["text"]:
        if isinstance(part, dict):
            full_text += part["text"]
        else:
            full_text += part
    return full_text

def process_data(ad_filepath, non_ad_filepath):
    ad_data = load_data(ad_filepath)
    non_ad_data = load_data(non_ad_filepath)

    ad_texts = [clean_text(extract_text(msg)) for msg in ad_data['messages'] if clean_text(extract_text(msg))]
    non_ad_texts = [clean_text(extract_text(msg)) for msg in non_ad_data['messages'] if clean_text(extract_text(msg))]

    texts = ad_texts + non_ad_texts
    labels = [1] * len(ad_texts) + [0] * len(non_ad_texts)

    return texts, labels

# Main code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ad_filepath = os.path.join(BASE_DIR, 'data/reklama', 'result.json')
non_ad_filepath = os.path.join(BASE_DIR, 'data/nereklama', 'result.json')
texts, labels = process_data(ad_filepath, non_ad_filepath)

# Load model and vectorizer
model = joblib.load('randfor_model.pkl')
vectorizer = joblib.load('randfor_vectorizer.pkl')

# Vectorize and prepare data
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
test_vectors = vectorizer.transform(test_texts)
feature_names = vectorizer.get_feature_names_out()
X_test = pd.DataFrame(test_vectors.toarray(), columns=feature_names)

# Initialize explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values - ensure we're using the correct class
shap_values = explainer.shap_values(X_test)

# Verify shapes match
print(f"SHAP values shape: {np.array(shap_values).shape}")
print(f"X_test shape: {X_test.shape}")

def save_shap_plot(plot_func, filename, *args, **kwargs):
    plt.figure()
    plot_func(*args, **kwargs)
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()

# Create output directory
output_dir = "shap_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Summary plot - make sure we're using the right class (usually index 1 for binary classification)
try:
    save_shap_plot(shap.summary_plot, 
                  f"{output_dir}/summary_plot.svg",
                  shap_values[1], X_test, show=False)
except IndexError:
    # If only one array is returned, use that
    save_shap_plot(shap.summary_plot, 
                  f"{output_dir}/summary_plot.svg",
                  shap_values, X_test, show=False)

# 2. Force plots (first 5 examples)
for i in range(min(5, len(X_test))):
    try:
        force_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        shap_vals = shap_values[1][i] if isinstance(shap_values, list) else shap_values[i]
    except IndexError:
        force_value = explainer.expected_value
        shap_vals = shap_values[i]
        
    force_plot = shap.force_plot(force_value,
                               shap_vals,
                               X_test.iloc[i,:],
                               matplotlib=True,
                               show=False)
    save_shap_plot(lambda: force_plot, 
                  f"{output_dir}/force_plot_{i}.svg")

# 3. Waterfall plots (first 5 examples)
for i in range(min(5, len(X_test))):
    try:
        exp_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        sh_vals = shap_values[1][i] if isinstance(shap_values, list) else shap_values[i]
    except IndexError:
        exp_value = explainer.expected_value
        sh_vals = shap_values[i]
        
    save_shap_plot(shap.plots.waterfall,
                  f"{output_dir}/waterfall_{i}.svg",
                  shap.Explanation(values=sh_vals,
                                  base_values=exp_value,
                                  data=X_test.iloc[i],
                                  feature_names=feature_names))

# 4. Decision plot
try:
    exp_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    sh_vals = shap_values[1][:100] if isinstance(shap_values, list) else shap_values[:100]
except IndexError:
    exp_value = explainer.expected_value
    sh_vals = shap_values[:100]

save_shap_plot(shap.decision_plot,
              f"{output_dir}/decision_plot.svg",
              exp_value,
              sh_vals,
              feature_names=feature_names,
              show=False)

print(f"Visualizations saved as SVG in {output_dir}")