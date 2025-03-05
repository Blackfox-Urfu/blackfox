import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Данные для разных процессоров
CPU0 = {
    "Intel Core i7-11700K": {"cost": 91500, "flops": 115.2, "power": 1542.86, "consumption": 287},
    "Intel Core i9-11900K": {"cost": 102500, "flops": 112, "power": 1500, "consumption": 387},
}

CPU1 = {
    "AMD Ryzen 5 5600X": {"cost": 91500, "flops": 88.8, "power": 1095.39, "consumption": 202},
    "AMD Ryzen 7 5800X": {"cost": 94500, "flops": 121.6, "power": 1573.3, "consumption": 242},
}

CPU2 = {
    "AMD Ryzen 5 5600X": {"cost": 91500, "flops": 88.8, "power": 1095.39, "consumption": 202},
    "AMD Ryzen 9 5900X": {"cost": 104500, "flops": 177.6, "power": 2846.15, "consumption": 242},
}

def linear_programming_example(users, CPU, max_consumption):
    cpu_models = list(CPU.keys())
    costs = [CPU[cpu]['cost'] for cpu in CPU]
    powers = [CPU[cpu]['power'] for cpu in CPU]
    consumptions = [CPU[cpu]['consumption'] for cpu in CPU]

    budget = 3000000

    c = costs
    A = [
        [-powers[0], -powers[1]],
        [consumptions[0], consumptions[1]],
        [costs[0], costs[1]]
    ]
    b = [-users, max_consumption, budget]
    bounds = [(0, None), (0, None)]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if result.success:
        x_rounded = np.ceil(result.x[0])
        y_rounded = np.ceil(result.x[1])
        plt.figure(figsize=(20, 16))

        x_max = max(max_consumption / consumptions[0] * 1.2, x_rounded * 1.5)
        x = np.linspace(0, x_max, 400)

        y1 = (max_consumption - consumptions[0] * x) / consumptions[1]
        y2 = (users - powers[0] * x) / powers[1] if powers[1] != 0 else np.full_like(x, np.inf)
        y3 = (budget - costs[0] * x) / costs[1] if costs[1] != 0 else np.full_like(x, np.inf)

        y_max = max(np.max(y1[y1 > 0]), np.max(y2[y2 > 0]), np.max(y3[y3 > 0]), y_rounded * 1.5)

        plt.plot(x, y1, 'r-', label=f'Ограничение энергопотребления: {consumptions[0]}x + {consumptions[1]}y ≤ {max_consumption} Вт')
        plt.plot(x, y2, 'b-', label=f'Требуемая производительность: {powers[0]:.1f}x + {powers[1]:.1f}y ≥ {users} пользователей')
        plt.plot(x, y3, 'g-', label=f'Ограничение бюджета: {costs[0]}x + {costs[1]}y ≤ {budget}')

        plt.plot(x_rounded, y_rounded, 'ro', markersize=10, label=f'Оптимальное решение: ({x_rounded}, {y_rounded})')
        plt.annotate(f'({x_rounded:.1f}, {y_rounded:.1f})',
                     xy=(x_rounded, y_rounded),
                     xytext=(10, 10),
                     textcoords='offset points')

        plt.xlim(0, x_max)
        plt.ylim(0, y_max)
        plt.xlabel(f'Количество {cpu_models[0]}', fontsize=12)
        plt.ylabel(f'Количество {cpu_models[1]}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.fill_between(x, np.maximum.reduce([y2, np.zeros_like(x)]), np.minimum.reduce([y1, y3]),
                 where=(np.minimum.reduce([y1, y3]) > np.maximum.reduce([y2, np.zeros_like(x)])) & (x <= x_max) & (np.minimum.reduce([y1, y3]) <= y_max),
                 color='green', alpha=0.3)

        plt.text(0.02, 0.98, 'Зеленая область - допустимые решения\nКрасная точка - оптимальное решение',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top', fontsize=12)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout(pad=3.0)

        plt.savefig(f'feasible_region_{cpu_models[0]}_{cpu_models[1]}_updated.png',
                    bbox_inches='tight', dpi=320, pad_inches=0.5)
    else:
        print("Linear programming failed:", result.message)
# Параметры
users = 40000
max_consumption = 8000

# Запуск
linear_programming_example(users, CPU0, max_consumption)
linear_programming_example(users, CPU1, max_consumption)
linear_programming_example(users, CPU2, max_consumption)