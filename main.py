"""
Виконав: Литвиненко Роман
Лабораторна робота №8, II рівень складності
Розробити програмний скрипт, що реалізує:
1. Скоринговий аналіз позичальників за даними Data_description.xlsx, Sample_data.xlsx відповідно до багатокритеріальної моделі.
2. Передбачити чи буде кредит повернено у форматі бінарної оцінки (0 або 1);
3. Виявлення шахрайства та фальсифікації даних.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
matplotlib.use('TkAgg')


def main():
    # Отримання даних
    data = pd.read_excel('sample_data.xlsx')
    scoring_data = pd.read_excel('coefs.xlsx')

    # Позбавлення від непотрібних даних
    necessary_columns = scoring_data['criteria'].dropna().tolist()
    filtered_data = data[necessary_columns]
    data_matrix = filtered_data.values.tolist()
    transposed_data_matrix = list(map(list, zip(*data_matrix)))

    # Обробка дат:
    #   якщо вік до 18 - 0,
    #   якщо від 18 до 30 - значення в проміжку [0, 1],
    #   якщо від 30 до 45 - значення 1,
    #   якщо від 45 до 60 - значення в проміжку [1, 0])
    #   якщо більше 60 - 0.
    for i, birthdate in enumerate(transposed_data_matrix[2]):
        years_old = 2024 - birthdate.year
        if 30 >= years_old > 18:
            transposed_data_matrix[2][i] = 1 / 12 * years_old - 18 / 12
        elif 45 >= years_old > 30:
            transposed_data_matrix[2][i] = 1
        elif 60 >= years_old > 45:
            transposed_data_matrix[2][i] = -1 / 15 * years_old + 4
        else:
            transposed_data_matrix[2][i] = 0

    # Нормалізація вагових коефіцієнтів
    weights = scoring_data['weight'].tolist()
    weights_normalized = []
    weights_sum = sum(weights)
    for weight in weights:
        weights_normalized.append(1 / weight / weights_sum)

    # Нормалізація критеріїв
    criteria_type = scoring_data['minmax'].tolist()
    criteria_matrix_normalized = []
    for i, criteria_row in enumerate(transposed_data_matrix):
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
            criteria_matrix_normalized[i].append((criteria_value if criteria_type[i] == 'min' else 1 / criteria_value) / criteria_sum)

    # Розрахунок інтегрованих оцінок
    integro = []
    for i in range(len(criteria_matrix_normalized[0])):
        integro.append(0)
        for j in range(len(criteria_matrix_normalized)):
            integro[i] += weights_normalized[j] * 1 / (1 - criteria_matrix_normalized[j][i])

    credit_scores = sorted(integro)

    # Вивід результатів скорингу:
    plt.plot(integro)
    plt.title('Integro')
    plt.show()
    good_cs_coutnter = 0
    for credit_score in credit_scores:
        if credit_score < 0.1017:
            good_cs_coutnter += 1
    plt.plot(credit_scores, label='Результати')
    plt.title('Кредитний скоринг')
    plt.hlines(0.102, 0, len(credit_scores), label=f'threshhold {good_cs_coutnter / len(integro):.3%}', color='orange')
    plt.legend()
    plt.show()

    # Пошук шахраїв к використанням моделей KNN, PCA та StandardScaler
    transposed_criteria_matrix_normalized = list(map(list, zip(*criteria_matrix_normalized)))
    outliers_fraction = 1 - good_cs_coutnter / len(integro)
    X = transposed_criteria_matrix_normalized
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)
    norm_data = StandardScaler().fit_transform(transposed_criteria_matrix_normalized)
    compressed = PCA(n_components=2).fit_transform(norm_data)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=compressed[:, 0], y=compressed[:, 1], hue=np.where(y_pred, "Шахрай", "Не шахрай"))
    plt.title('Пошук шахраїв')
    plt.show()

    print('Бінарна оцінка повернення кредиту:')
    print(y_pred)


if __name__ == '__main__':
    main()
