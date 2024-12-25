"""
Виконав: Литвиненко Роман
МКР: Реалізувати скрипт із визначенням інтегрованої оцінки ефективності 10 товарів за 3 показниками ефективності.

"""

import pandas as pd


def main():
    # Читання даних з файлу
    weights, criteria_names, criteria_type, restaurants_names, criteria_matrix = read_data_from_xlsx('data.xlsx')

    # Нормалізація вагових коефіцієнтів
    weights_normalized = []
    weights_sum = sum(weights)
    for weight in weights:
        weights_normalized.append(1 / weight / weights_sum)

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
            integro[i] += (1 - weights_normalized[j]) * 1 / (1 - criteria_matrix_normalized[j][i])

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
    print(f'Оптимальний товар: {restaurants_names[optimum]}')


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


if __name__ == '__main__':
    main()