"""
Виконав: Литвиненко Роман
Лабораторна робота №3, II рівень складності + додаткове завдання

"""

import pandas as pd
from ortools.sat.python import cp_model


def main():
    # Читання даних з файлу
    weights, criteria_names, criteria_type, restaurants_names, criteria_matrix = read_data_from_xlsx('restaurants.xlsx')

    # Нормалізація вагових коефіцієнтів
    weights_normalized = []
    weights_sum = sum(weights)
    for weight in weights:
        weights_normalized.append(1/weight/weights_sum)

    # Нормалізація критеріїв
    criteria_matrix_normalized = []
    for i, criteria_row in enumerate(criteria_matrix):
        for j, criteria_value in enumerate(criteria_row):
            if criteria_value == 0:
                criteria_row[j] = 0.0000000000000000000001
        criteria_sum = 0
        criteria_matrix_normalized.append([])
        if criteria_type[i] == 'min':
            criteria_sum = sum(criteria_row)
        else:
            for criteria_value in criteria_row:
                criteria_sum += 1 / criteria_value
        for j, criteria_value in enumerate(criteria_row):
            criteria_matrix_normalized[i].append((criteria_value if criteria_type[i] == 'min' else 1 / criteria_value)
                                                 / criteria_sum)

    # Розрахунок інтегрованих оцінок
    integro = []
    for i in range(len(criteria_matrix_normalized[0])):
        integro.append(0)
        for j in range(len(criteria_matrix_normalized)):
            integro[i] += weights_normalized[i] * 1 / (1 - criteria_matrix_normalized[j][i])

    # Пошук оптимуму
    minimal = float('Infinity')
    optimum = 0
    for i in range(len(integro)):
        if minimal > integro[i]:
            minimal = integro[i]
            optimum = i

    # Вивід результатів багатокритеріального оцінювання ефективності
    print('Інтегрована оцінка:')
    for i, restaurant in enumerate(restaurants_names):
        print(f'{integro[i]:.5f} - {restaurant}')
    print(f'Оптимальний ресторан: {restaurants_names[optimum]}')


# Функція парсингу даних з файлу за особливою структурою
def read_data_from_xlsx(path):
    print(f'Парсинг з файлу {path}...')
    raw_data = pd.read_excel(path)
    weights = raw_data['Вагові коефіцієнти'].tolist()
    criteria_names = raw_data['Критерії'].values.tolist()
    criteria_type = raw_data['Тип'].values.tolist()
    restaurants_names = raw_data.columns[3:].tolist()
    criteria_matrix = raw_data.iloc[:, 3:].values.tolist()
    return weights, criteria_names, criteria_type, restaurants_names, criteria_matrix


# Додаткове завдання,розв’язок задачі лінійного програмування для умов, зазначених в Лекції_6 з використанням
# інструментів бібліотеки Google OR-Tools
def additional_task():
    # Оптимізаційна модель
    model = cp_model.CpModel()

    # Змінні з верхньою межею для кожної з них
    var_upper_bound = 10
    X1 = model.NewIntVar(0, var_upper_bound, 'X1')
    X2 = model.NewIntVar(0, var_upper_bound, 'X2')
    X3 = model.NewIntVar(0, var_upper_bound, 'X3')
    X4 = model.NewIntVar(0, var_upper_bound, 'X4')
    X5 = model.NewIntVar(0, var_upper_bound, 'X5')
    X6 = model.NewIntVar(0, var_upper_bound, 'X6')

    # Обмеження
    model.Add(3 * X1 + 4 * X2 - 2 * X4 <= 24)
    model.Add(X1 + 2 * X2 - X3 <= 8)
    model.Add(4 * X1 - X5 <= 16)
    model.Add(4 * X2 - X6 <= 12)

    # Цільова функція ефективності
    model.Minimize(-2 * X1 - 2 * X2)

    # Вирішувач
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print(f'Результат мінімізованого значення: {solver.ObjectiveValue()}')
        X1_out = solver.Value(X1)
        X2_out = solver.Value(X2)
        X3_out = solver.Value(X3)
        X4_out = solver.Value(X4)
        X5_out = solver.Value(X5)
        X6_out = solver.Value(X6)
        print('X1= ', X1_out)
        print('X2= ', X2_out)
        print('X3= ', X3_out)
        print('X4= ', X4_out)
        print('X5= ', X5_out)
        print('X6= ', X6_out)



# Оптимізаційна модель
model = cp_model.CpModel()

# Змінні з верхньою межею для кожної з них
var_upper_bound = 10000
X1 = model.NewIntVar(0, var_upper_bound, 'X1')
X2 = model.NewIntVar(0, var_upper_bound, 'X2')
X3 = model.NewIntVar(0, var_upper_bound, 'X3')

# Обмеження
model.Add(10 * X1 + 20 * X2 + 3 * X3 <= 1000)
model.Add(5 * X1 + 4 * X2 + 1 * X3 <= 500)
model.Add(X1 >= 50)
model.Add(X2 >= 10)
model.Add(X3 >= 11)

# Цільова функція ефективності
model.Minimize(10 * X1 + 20 * X2 + 3 * X3)

# Вирішувач
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print(f'Результат мінімізованого значення: {solver.ObjectiveValue()}')
    X1_out = solver.Value(X1)
    X2_out = solver.Value(X2)
    X3_out = solver.Value(X3)
    print('X1= ', X1_out)
    print('X2= ', X2_out)
    print('X3= ', X3_out)


# Вхід у програму
if __name__ == '__main__':
    main()
    print('----------')
    additional_task()

