import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

# Загрузка модели и векторизатора
rf_model = joblib.load('randfor_best_model.pkl')
vectorizer = joblib.load('randfor_vectorizer.pkl')

# Выбираем одно дерево из случайного леса
single_tree = rf_model.estimators_[0]

# Ограничиваем глубину дерева для визуализации
max_depth = 12

# Огромный холст для улучшенной читаемости
plt.figure(figsize=(50, 25))  # Уменьшите размеры для удобства
font_size = 12  

# Визуализация дерева с ограниченной глубиной
plot_tree(
    single_tree,
    filled=True,
    feature_names=vectorizer.get_feature_names_out(),
    max_depth=max_depth,  # Ограничение глубины
    fontsize=font_size,
    proportion=True  # Узлы одинакового размера
)

# Сохраняем в формате SVG
plt.savefig(
    "decision_tree_limited_depth.svg",  # Имя выходного файла
    format="svg",
    dpi=600
)


