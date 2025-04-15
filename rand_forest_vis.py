import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
import csv
from sklearn.ensemble import RandomForestClassifier

# Загрузка модели и векторизатора
rf_model = joblib.load('randfor_model.pkl')
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

# Открытие CSV-файла для записи
with open('random_forest_structure.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Запись заголовков
    writer.writerow(['Tree Index', 'Node Index', 'Feature', 'Threshold', 'Left Child', 'Right Child', 'Impurity', 'Samples'])

    # Итерация по всем деревьям в случайном лесе
    for tree_index, tree in enumerate(rf_model.estimators_):
        tree_ = tree.tree_
        for node_index in range(tree_.node_count):
            feature = tree_.feature[node_index]
            threshold = tree_.threshold[node_index]
            left_child = tree_.children_left[node_index]
            right_child = tree_.children_right[node_index]
            impurity = tree_.impurity[node_index]
            samples = tree_.n_node_samples[node_index]

            # Запись информации о каждом узле
            writer.writerow([
                tree_index,
                node_index,
                feature,
                threshold,
                left_child,
                right_child,
                impurity,
                samples
            ])

print("Structure of all trees in the random forest has been saved to 'random_forest_structure.csv'.")


