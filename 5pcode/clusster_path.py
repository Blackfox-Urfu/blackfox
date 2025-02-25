import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import random
from sklearn.tree import export_graphviz
import graphviz
from scipy.optimize import linprog
import pandas as pd
import json
# Обновленные данные конфигурации CPU
CPU = {
    "AMD Ryzen 5 5600X": {"cost": 91500, "flops": 88.8, "power": 1095.39, "consumption": 202},
    "AMD Ryzen 7 5800X": {"cost": 94500, "flops": 121.6, "power": 1027.03, "consumption": 242},
    "AMD Ryzen 9 5900X": {"cost": 104500, "flops": 177.6, "power": 2846.15, "consumption": 242},
    "Intel Core i5-11600K": {"cost": 89500, "flops": 93.6, "power": 1218.75, "consumption": 262},
    "Intel Core i7-11700K": {"cost": 91500, "flops": 115.2, "power": 1542.86, "consumption": 287},
    "Intel Core i9-11900K": {"cost": 102500, "flops": 112, "power": 1500, "consumption": 387},
}

def fluctuate_price(base_price, fluctuation_rate=0.01):
    return base_price * (1 + random.uniform(-fluctuation_rate, fluctuation_rate))

def fluctuate_users(base_users, fluctuation_rate=0.05):
    return int(base_users * (1 + random.uniform(-fluctuation_rate, fluctuation_rate)))

def calculate_profit(users, power_per_server, server_price, resale_value, cluster_changed):
    """
    Расчет прибыли, окупаемости, мощности и стоимости серверов.
    """
    servers_needed = np.ceil(users / power_per_server)
    revenue = users * 89
    Variable_costs = 0.0023 * 720 * 7 * users
    OPEX = 10000
    CAPEX = servers_needed * server_price if cluster_changed else 0
    profit = revenue - (OPEX + CAPEX + Variable_costs)
    total_power = power_per_server * servers_needed
    total_cost = server_price * servers_needed
    return profit, servers_needed, total_power, total_cost, revenue, CAPEX, Variable_costs, OPEX

def optimize_servers(users, CPU):
    """
    Оптимизация количества серверов с использованием линейного программирования.
    """
    costs = [data['cost'] for data in CPU.values()]
    powers = [data['power'] for data in CPU.values()]
    consumptions = [data['consumption'] for data in CPU.values()]

    # Целевая функция: максимизация прибыли
    # Прибыль = Выручка - Затраты
    # Затраты = Стоимость серверов + Переменные затраты + OPEX
    # Выручка = Пользователи * 89
    revenue_per_user = 89
    variable_cost_per_user = 0.0023 * 720 * 7
    opex = 1770

    # Прибыль на сервер = (Выручка - Затраты) / Количество серверов
    profit_per_server = [revenue_per_user - (costs[i] + variable_cost_per_user + opex) for i in range(len(CPU))]

    # Преобразуем задачу максимизации в задачу минимизации
    c = [-p for p in profit_per_server]

    # Ограничения: суммарная мощность >= требуемой мощности и суммарное энергопотребление <= 10000
    A = [-np.array(powers), np.array(consumptions)]
    b = [-users, 10000]

    # Ограничения на количество серверов (целочисленные)
    bounds = [(0, None) for _ in range(len(powers))]

    # Решение задачи линейного программирования
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if result.success:
        return np.ceil(result.x)  # Округление до целого числа серверов
    else:
        raise ValueError("Не удалось найти оптимальное решение")

def find_best_path(CPU, months, users_per_month_increase, discount=0.2):
    """
    Нахождение оптимальной конфигурации серверов.
    """
    best_path = []
    current_config = None
    total_profit = 0

    # Открытие файла для записи с указанием кодировки utf-8
    with open('calculations_description.txt', 'w', encoding='utf-8') as file:
        file.write("Описание вычислений с подставленными значениями для всех месяцев:\n")

        # Добавление общего прогресс-бара
        for month in tqdm(range(1, months + 1), desc="Processing months"):
            users = fluctuate_users(month * users_per_month_increase)
            best_profit = -float('inf')
            best_config = {}

            # Оптимизация серверов
            try:
                server_counts = optimize_servers(users, CPU)
            except ValueError as e:
                print(f"Month {month}: {e}")
                continue

            total_power = sum(server_counts[i] * list(CPU.values())[i]['power'] for i in range(len(CPU)))
            total_price = sum(server_counts[i] * fluctuate_price(list(CPU.values())[i]['cost']) for i in range(len(CPU)))

            cluster = [{"server": list(CPU.keys())[i], "count": int(server_counts[i]), "power": list(CPU.values())[i]['power'], "price": list(CPU.values())[i]['cost']} for i in range(len(CPU)) if server_counts[i] > 0]

            cluster_changed = json.dumps(current_config, sort_keys=True) != json.dumps(cluster, sort_keys=True)
            profit, servers_needed, _, _, revenue, CAPEX, Variable_costs, OPEX = calculate_profit(users, total_power, total_price, 0, cluster_changed)

            if profit > best_profit:
                best_profit = profit
                best_config = {
                    "month": month,
                    "users": users,
                    "cluster": cluster,
                    "profit_per_month": profit,
                    "servers_needed": sum(server_counts),  # Суммируем количество серверов
                    "total_power": total_power,
                    "total_cost": total_price,
                    "revenue": revenue,
                    "CAPEX": CAPEX,
                    "Variable_costs": Variable_costs,
                    "OPEX": OPEX,
                    "total_spent": CAPEX + Variable_costs + OPEX
                }

            best_path.append(best_config)
            current_config = best_config['cluster']
            total_profit += best_profit
            best_config["profit_total"] = total_profit

            # Запись в файл
            file.write(f"Месяц {month}:\n")
            file.write(f"- Пользователи: {best_config['users']}\n")
            file.write(f"- Выручка: {best_config['revenue']:.2f} (Формула: Пользователи * 89 = {best_config['users']} * 89)\n")
            file.write(f"- Переменные затраты: {best_config['Variable_costs']:.2f} (Формула: 0.0023 * 720 * 7 * {best_config['users']})\n")
            file.write(f"- OPEX: {best_config['OPEX']:.2f} (Фиксированные затраты)\n")
            file.write(f"- CAPEX: {best_config['CAPEX']:.2f}\n")
            file.write(f"- Прибыль за месяц: {best_config['profit_per_month']:.2f}\n")
            file.write(f"- Накопленная прибыль: {best_config['profit_total']:.2f}\n")
            if cluster_changed:
                file.write(f"- Принятые решения: {best_config['cluster']}\n\n")
            else:
                file.write("- Принятые решения: Без изменений\n\n")

            print(f"Month {month}: Best profit {best_profit}")

    # Сохранение результатов
    df_best_path = pd.DataFrame(best_path)
    # Выявление изменений в 'cluster'
    df_best_path['cluster_changed'] = df_best_path['cluster'] != df_best_path['cluster'].shift()
    # Фильтрация изменений
    changes = df_best_path[df_best_path['cluster_changed']][['month', 'cluster']]
    pd.set_option('display.max_colwidth', None)  # Убрать ограничение на длину строки
    pd.set_option('display.max_rows', None)  # Убрать ограничение на количество строк
    pd.set_option('display.max_columns', None)  # Убрать ограничение на количество колонок
    print(changes)
    excel_file = 'best_path_optimization.xlsx'
    df_best_path.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"Наилучший путь сохранен в файл {excel_file}.")

    # Построение графика области допустимых значений и сохранение в файл
    plt.figure(figsize=(12, 8))
    plt.plot(df_best_path['month'], df_best_path['servers_needed'], label='Количество серверов')
    plt.plot(df_best_path['month'], df_best_path['profit_per_month'], label='Прибыль за месяц')
    plt.xlabel('Месяц')
    plt.ylabel('Значение')
    plt.title('Область допустимых значений для управляющих переменных')
    plt.legend()
    plt.grid(True)
    plt.savefig('feasible_region.png')

    # Построение графика финансовых показателей и сохранение в файл
    plt.figure(figsize=(12, 8))
    plt.plot(df_best_path['month'], df_best_path['profit_per_month'], label='Прибыль за месяц')
    plt.plot(df_best_path['month'], df_best_path['CAPEX'], label='CAPEX')
    plt.plot(df_best_path['month'], df_best_path['Variable_costs'], label='Переменные затраты')
    plt.plot(df_best_path['month'], df_best_path['total_spent'], label='Совокупные затраты')
    plt.xlabel('Месяц')
    plt.ylabel('Финансовые показатели')
    plt.title('Финансовые показатели по месяцам')
    plt.legend()
    plt.grid(True)
    plt.savefig('financial_indicators.png')

    return df_best_path

# Параметры для расчета
months = 120
users_per_month_increase = 1000

# Запуск оптимизации
find_best_path(CPU, months, users_per_month_increase)

