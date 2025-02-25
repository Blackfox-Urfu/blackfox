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

def linear_programming_example(users, CPU, max_consumption=1000):
    """
    Решает задачу линейного программирования и строит график.
    """
    cpu_models = list(CPU.keys())
    costs = [CPU[cpu]['cost'] for cpu in CPU]
    powers = [CPU[cpu]['power'] for cpu in CPU]
    consumptions = [CPU[cpu]['consumption'] for cpu in CPU]
    
    c = costs
    A = [[-powers[0], -powers[1]],
         [consumptions[0], consumptions[1]]]
    b = [-users, max_consumption]
    bounds = [(0, None), (0, None)]
    
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if result.success:
        # Округляем решения до целых чисел
        x_rounded = np.ceil(result.x[0])
        y_rounded = np.ceil(result.x[1])

        x = np.linspace(0, max_consumption / consumptions[0], 400)
        y1 = (max_consumption - consumptions[0] * x) / consumptions[1]
        y2 = (users - powers[0] * x) / powers[1] if powers[1] != 0 else 0

        plt.figure(figsize=(20, 14))
        plt.plot(x, y1, 'r-', label=f'Ограничение энергопотребления: {consumptions[0]}x + {consumptions[1]}y ≤ {max_consumption} Вт')
        plt.plot(x, y2, 'b-', label=f'Требуемая производительность: {powers[0]:.1f}x + {powers[1]:.1f}y ≥ {users} пользователей')
        plt.plot(x_rounded, y_rounded, 'ro', markersize=10, label=f'Оптимальное решение: ({x_rounded}, {y_rounded})')
        plt.annotate(f'({x_rounded:.1f}, {y_rounded:.1f})',
                     xy=(x_rounded, y_rounded),
                     xytext=(10, 10),
                     textcoords='offset points')
        
        plt.xlim(0, max_consumption / consumptions[0])
        plt.ylim(0, users / powers[1] if powers[1] != 0 else 10)
        plt.xlabel(f'Количество {cpu_models[0]}')
        plt.ylabel(f'Количество {cpu_models[1]}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.fill_between(x, np.maximum(y2, 0), y1, where=y1 > np.maximum(y2, 0), color='green', alpha=0.3, label='Допустимая область')
        plt.text(0.02, 0.98, 'Зеленая область - допустимые решения\nКрасная точка - оптимальное решение',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
        plt.tight_layout()
        plt.savefig(f'feasible_region_{cpu_models[0]}_{cpu_models[1]}.png', bbox_inches='tight', dpi=320)
        plt.show()
    else:
        print("Linear programming failed:", result.message)

# Параметры
users = 40000
max_consumption = 10000

# Запуск
linear_programming_example(users, CPU0, max_consumption)
linear_programming_example(users, CPU1, max_consumption)
linear_programming_example(users, CPU2, max_consumption)
