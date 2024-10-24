"""
Виконав: Литвиненко Роман
Лабораторна робота №3, II рівень складності + додаткове завдання

"""

#----------------------------- Decision Support System (DSS) -------------------------------------

'''

Завдання:
   сформувати обгрунтоване рішення щодо впровадження нового товару на ринок з множини альтернатив
   за багатокритеріальною / багатофакторною оптимізаційною моделлю для Decision Support System: ERP,CRM систем.

Склад етапів:
1. Формалізація задачі як багатофакторної / багатокритеріальної;
2. Формування сегменту даних та парсінг *.xls файлу;
3. Нормалізація даних;
4. Розрахунок інтегрованої оцінки - scor - оцінювання альтернатив - scoring;
   спосіб розрахунку інтегрованої оцінки - нелінійна схема компромісів
   http://sci-gems.math.bas.bg/jspui/bitstream/10525/49/1/ijita15-2-p02.pdf
5. Обрання найкращіх альтернатив.

Альтернативні галузі:
   Теорія операцій - розподіл ресурсів;
   Теорія розкладів
   Інструментарій оптимізації: Google OR-Tools;
   https://developers.google.com/optimization/examples?hl=en

'''


import pandas as pd
import numpy as np


def main():
    weights, criteria_names, criteria_type, restaurants_names, criteria_matrix = read_data_from_xlsx('restaurants.xlsx')

    weights_normalized = []
    weights_sum = sum(weights)
    # Нормалізація вагових коефіцієнтів
    for weight in weights:
        weights_normalized.append(weight/weights_sum)

    criteria_matrix_normalized = []
    for i, criteria_row in enumerate(criteria_matrix):
        for j, criteria_value in enumerate(criteria_row):
            if criteria_value == 0:
                criteria_row[j] = 0.0000000000000000000001
        criteria_sum = 0
        criteria_matrix_normalized.append([])
        if criteria_type[i] == 'min':
            print('min')
            criteria_sum = sum(criteria_row)
        else:
            print('max')
            for criteria_value in criteria_row:
                criteria_sum += 1 / criteria_value
        for j, criteria_value in enumerate(criteria_row):
            criteria_matrix_normalized[i].append((criteria_value if criteria_type[i] == 'min' else 1 / criteria_value) / criteria_sum)
        print(criteria_matrix_normalized[i])

    integro = []
    for i in range(len(criteria_matrix_normalized[0])):
        integro.append(0)
        for j in range(len(criteria_matrix_normalized)):
            integro[i] += weights_normalized[i] * 1 / (1 - criteria_matrix_normalized[j][i])

    # criteria_matrix_normalized = []
    # for i, criteria_row in enumerate(criteria_matrix):
    #     criteria_matrix_normalized.append([])
    #     criteria_row_buff = criteria_row.copy()
    #     for j, criteria_value in enumerate(criteria_row_buff):
    #         if criteria_value == 0:
    #             criteria_value = 0.0000000000000001
    #         if criteria_type[i] == 'min':
    #             criteria_row_buff[j] = criteria_value
    #         else:
    #             criteria_row_buff[j
    #             ] = 1 / criteria_value
    #     criteria_sum = sum(criteria_row)
    #     for criteria_value in criteria_row_buff:
    #         criteria_matrix_normalized[i].append(criteria_value / criteria_sum)
    #     print(i, criteria_matrix_normalized[i])

    # integro = []
    # for i in range(len(criteria_matrix_normalized[0])):
    #     integro.append(0)
    #     # print(0)
    #     for j in range(len(criteria_matrix_normalized)):
    #         print(i, j, 1 / (1 - criteria_matrix_normalized[j][i]))
    #
    #         # print(f'+ {weights_normalized[j]} * (1 - {criteria_matrix_normalized[j][i]}) ** (-1) == {weights_normalized[j] * (1 - criteria_matrix_normalized[j][i]) ** (-1)}')
    #         integro[i] += weights_normalized[j] * 1 / (1 - criteria_matrix_normalized[j][i])
    #     print(integro[i])
    #     print('-----')

    # Пошук оптимуму
    minimal = float('Infinity')
    optimum = 0
    for i in range(len(restaurants_names)):
        if minimal > integro[i]:
            minimal = integro[i]
            optimum = i

    print('Інтегрована оцінка:')
    for i, restaurant in enumerate(restaurants_names):
        print(f'{integro[i]:.4f} - {restaurant}')
    print(f'Оптимальний ресторан: {restaurants_names[optimum]}')



# Функція парсингу
def read_data_from_xlsx(path):
    print(f'Парсинг з файлу {path}...')
    raw_data = pd.read_excel(path)
    weights = raw_data['Вагові коефіцієнти'].tolist()
    criteria_names = raw_data['Критерії'].values.tolist()
    criteria_type = raw_data['Тип'].values.tolist()
    restaurants_names = raw_data.columns[3:].tolist()
    criteria_matrix = raw_data.iloc[:, 3:].values.tolist()
    return weights, criteria_names, criteria_type, restaurants_names, criteria_matrix




def Voronin(File_name, G1, G2, G3, G4, G5, G6, G7, G8, G9):

    # --------------------- вхідні дані -------------------------
    line_column_matrix = matrix_generation(File_name)
    print(line_column_matrix)
    column_matrix = np.shape(line_column_matrix)
    print(column_matrix)
    Integro = np.zeros((column_matrix[1]))

    F1 = matrix_adapter(line_column_matrix, 0)
    F2 = matrix_adapter(line_column_matrix, 1)
    F3 = matrix_adapter(line_column_matrix, 2)
    F4 = matrix_adapter(line_column_matrix, 3)
    F5 = matrix_adapter(line_column_matrix, 4)
    F6 = matrix_adapter(line_column_matrix, 5)
    F7 = matrix_adapter(line_column_matrix, 6)
    F8 = matrix_adapter(line_column_matrix, 7)
    F9 = matrix_adapter(line_column_matrix, 8)

    print(F1)
    print(F2)

    #--------------- нормалізація вхідних даних ------------------
    F10 = np.zeros((column_matrix[1]))
    F20 = np.zeros((column_matrix[1]))
    F30 = np.zeros((column_matrix[1]))
    F40 = np.zeros((column_matrix[1]))
    F50 = np.zeros((column_matrix[1]))
    F60 = np.zeros((column_matrix[1]))
    F70 = np.zeros((column_matrix[1]))
    F80 = np.zeros((column_matrix[1]))
    F90 = np.zeros((column_matrix[1]))

    GNorm = G1 + G2 + G3 + G4 + G5 + G6 + G6 + G7 + G8 + G9
    G10 = G1 / GNorm
    G20 = G2 / GNorm
    G30 = G3 / GNorm
    G40 = G4 / GNorm
    G50 = G5 / GNorm
    G60 = G6 / GNorm
    G70 = G7 / GNorm
    G80 = G8 / GNorm
    G90 = G9 / GNorm

    sum_F1=sum_F2=sum_F3=sum_F4=sum_F5=sum_F6=sum_F7=sum_F8=sum_F9 = 0

    for i in range(column_matrix[1]):
        sum_F1 = sum_F1 + F1[i]
        sum_F2 = sum_F2 + F2[i]
        sum_F3 = sum_F3 + F3[i]
        sum_F4 = sum_F4 + F4[i]
        sum_F5 = sum_F5 + F5[i]
        sum_F6 = sum_F6 + (1 / F6[i])  # максимізований критерії
        sum_F7 = sum_F7 + F7[i]
        sum_F8 = sum_F8 + F8[i]
        sum_F9 = sum_F9 + F9[i]

    for i in range(column_matrix[1]):
        # --------------- нормалізація критеріїв ------------------
        F10[i] = F1[i] / sum_F1
        F20[i] = F2[i] / sum_F2
        F30[i] = F3[i] / sum_F3
        F40[i] = F4[i] / sum_F4
        F50[i] = F5[i] / sum_F5
        F60[i] = (1/F6[i]) / sum_F6  # максимізований критерії
        F70[i] = F7[i] / sum_F7
        F80[i] = F8[i] / sum_F8
        F90[i] = F9[i] / sum_F9

        Integro[i] = (G10*(1 - F10[i]) ** (-1))  + (G20*(1 - F20[i]) ** (-1)) + (G30*(1 - F30[i]) ** (-1))
        + (G40 * (1 - F40[i]) ** (-1)) + (G50 * (1 - F50[i]) ** (-1)) + (G60 * (1 - F60[i]) ** (-1))
        + (G70*(1 - F70[i]) ** (-1))  + (G80*(1 - F80[i]) ** (-1)) + (G90*(1 - F90[i]) ** (-1))

    # --------------- генерація оптимального рішення ----------------
    min=10000
    opt=0
    for i in range(column_matrix[1]):
        if min > Integro[i]:
            min = Integro[i]
            opt=i
    print('Інтегрована оцінка (scor):')
    print(Integro)
    print('Номер_оптимального_товару:', opt)

    return


# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ------------------------------
if __name__ == '__main__':
    main()
    # File_name = 'Pr1.xls'
    #
    # # ---------------- коефіціенти переваги критеріїв -----------------
    # G1 = G2 = G3 = G4 = G5 = G6 = G7 = G8 = G9 = 1
    # G1 = 1           # коефіціент домінування критерію
    #
    # Voronin(File_name, G1, G2, G3, G4, G5, G6, G7, G8, G9)

