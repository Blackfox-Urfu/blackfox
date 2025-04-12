import sharp
import os 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ad_filepath = os.path.join(BASE_DIR, 'data/reklama', 'result.json')
non_ad_filepath = os.path.join(BASE_DIR, 'data/nereklama', 'result.json')
texts, labels = process_data(ad_filepath, non_ad_filepath)


train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
test_vectors = vectorizer.transform(test_texts)

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)


# Visualize importance of features
shap.summary_plot(shap_values, X_test)

# Почему модель сделала это предсказание?
shap.initjs() # Включить интерактивную визуализацию 
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Водопадная визуализация
shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test.iloc[0]))

