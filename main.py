"""
Виконав: Литвиненко Роман
Лабораторна робота №7, II рівень складності
Варінт: 7. Розробити програмний скрипт, що реалізує аналіз даних, поданих у файлі Data_Set_7.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def main():
    page2_content = pd.read_excel('Data_Set_7.xlsx', sheet_name='qrySales')
    page2_content['OrderDate'] = pd.to_datetime(page2_content['OrderDate'], format='%d.%m.%Y')

    data_sorted_by_date = page2_content.sort_values(by='OrderDate')
    time_row_quantity = []
    time_row_revenue = []
    time_row_quantity_bypaid = [[], []]
    time_row_revenue_bypaid = [[], []]
    previous_date = None
    for i, row in data_sorted_by_date.iterrows():
        row = row.to_dict()
        current_date = row['OrderDate']
        if previous_date is None or previous_date != current_date:
            time_row_quantity.append(row['Quantity'])
            time_row_revenue.append(row['Revenue'])
            if row['Paid?'] == 'Yes':
                time_row_quantity_bypaid[0].append(row['Quantity'])
                time_row_revenue_bypaid[0].append(row['Revenue'])
                time_row_quantity_bypaid[1].append(0)
                time_row_revenue_bypaid[1].append(0)
            else:
                time_row_quantity_bypaid[1].append(row['Quantity'])
                time_row_revenue_bypaid[1].append(row['Revenue'])
                time_row_quantity_bypaid[0].append(0)
                time_row_revenue_bypaid[0].append(0)
        else:
            time_row_quantity[-1] += row['Quantity']
            time_row_revenue[-1] += row['Revenue']
            if row['Paid?'] == 'Yes':
                time_row_quantity_bypaid[0][-1] += row['Quantity']
                time_row_revenue_bypaid[0][-1] += row['Revenue']
            else:
                time_row_quantity_bypaid[1][-1] += row['Quantity']
                time_row_revenue_bypaid[1][-1] += row['Revenue']
        previous_date = row['OrderDate']

    print(time_row_quantity_bypaid)
    print(time_row_revenue_bypaid)

    print(time_row_quantity)
    print(time_row_revenue)
    # Вивід графіку кількості продажів
    plt.plot(time_row_quantity)
    plt.title('Quantity by days')
    plt.show()
    # Вивід графіку виручки
    plt.plot(time_row_revenue)
    plt.title('Revenue by days')
    plt.show()
    # Вивід обох графіків на одному холсті
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('days')
    ax1.set_ylabel('Quantity', color='tab:orange')
    ax1.plot(time_row_quantity, color='tab:orange')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Revenue', color='tab:blue')
    ax2.plot(time_row_revenue, color='tab:blue', linestyle='dashed')
    fig.tight_layout()
    plt.show()
    # Вивід графіку кількості продажів з візуалізацією оплати
    plt.plot(time_row_quantity_bypaid[0], label='paid')
    plt.plot(time_row_quantity_bypaid[1], label='not paid')
    plt.title('Quantity by days splitted by pay status')
    plt.legend()
    plt.show()
    # Вивід графіку виручки з візуалізацією оплати
    plt.plot(time_row_revenue_bypaid[0], label='paid')
    plt.plot(time_row_revenue_bypaid[1], label='not paid')
    plt.title('Revenue by days splitted by pay status')
    plt.legend()
    plt.show()


    months_data = {
        2008: {},
        2009: {}
    }
    for i, row in data_sorted_by_date.iterrows():
        row = row.to_dict()
        month = row['OrderDate'].month
        year = row['OrderDate'].year
        if month in months_data[year].keys():
            months_data[year][month][0].append(row['Quantity'])
            months_data[year][month][1].append(row['Revenue'])
        else:
            months_data[year][month] = [[row['Quantity']], [row['Revenue']]]
    for i in range(1, 10):
        if i not in months_data[2008].keys():
            months_data[2008][i] = [[0], [0]]
        if i not in months_data[2009].keys():
            months_data[2009][i] = [[0], [0]]

    months_2008 = [[], []]
    months_2009 = [[], []]
    for key in sorted(months_data[2008]):
        months_2008[0].append(sum(months_data[2008][key][0]))
        months_2008[1].append(sum(months_data[2008][key][1]))
    for key in sorted(months_data[2009]):
        months_2009[0].append(sum(months_data[2009][key][0]))
        months_2009[1].append(sum(months_data[2009][key][1]))

    # Гістограми виручки та кількості продаж за 2008 та 2009
    month_labels = [f'Month {i + 1}' for i in range(10)]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(month_labels))  # X positions for months
    bar_width = 0.4
    ax[0].bar(x - bar_width / 2, months_2008[0], bar_width, label='2008', color='skyblue')
    ax[0].bar(x + bar_width / 2, months_2009[0], bar_width, label='2009', color='orange')
    ax[0].set_ylabel('Products Count')
    ax[0].set_title('Monthly Products Count')
    ax[0].legend()
    ax[1].bar(x - bar_width / 2, months_2008[1], bar_width, label='2008', color='skyblue')
    ax[1].bar(x + bar_width / 2, months_2009[1], bar_width, label='2009', color='orange')
    ax[1].set_ylabel('Revenue')
    ax[1].set_title('Monthly Revenue')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(month_labels)
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    for i in range(len(months_data[2008])):
        print(months_data[2008][i + 1][0])
        plt.plot(months_data[2008][i + 1][0], label=f'month_{i+1}')
    plt.title('Quantity by months 2008')
    plt.legend()
    plt.show()

    for i in range(len(months_data[2008])):
        print(months_data[2008][i + 1][0])
        plt.plot(months_data[2008][i + 1][0], label=f'month_{i+1}')
    plt.title('Quantity by months 2008')
    plt.legend()
    plt.show()

    for i in range(len(months_data[2008])):
        print(months_data[2008][i + 1][1])
        plt.plot(months_data[2008][i + 1][1], label=f'month_{i+1}')
    plt.title('Revenue by months 2008')
    plt.legend()
    plt.show()

    plt.plot(months_2008[1], label='2008', marker='o')
    plt.plot(months_2009[1], label='2009', marker='o')
    plt.title('Revenue by months 2008-2009')
    plt.legend()
    plt.show()

    all_months_quantity = months_2008[0] + months_2009[0]
    all_months_revenue = months_2008[1] + months_2009[1]

    mnk_quantity = create_fourth_poly_model_mnk(np.arange(0, len(time_row_quantity)), time_row_quantity, 'МНК за кількістю проданого товару')
    stat_characteristics(time_row_quantity, mnk_quantity, 'Кількісті проданого товару')
    mnk_revenue = create_fourth_poly_model_mnk(np.arange(0, len(time_row_revenue)), time_row_revenue, 'МНК за виручкою')
    stat_characteristics(time_row_revenue, mnk_revenue, 'Виручки')
    mnk_quantity_by_months = create_fourth_poly_model_mnk(np.arange(0, len(all_months_quantity)), all_months_quantity, 'МНК за кількістю проданого товару по місяцям')
    stat_characteristics(all_months_quantity, mnk_quantity_by_months, 'Кількісті проданого товару за місяцями')
    mnk_revenue_by_months = create_fourth_poly_model_mnk(np.arange(0, len(all_months_revenue)), all_months_revenue, 'МНК за виручкою по місяцям')
    stat_characteristics(all_months_revenue, mnk_revenue_by_months, 'Виручки за місяцями')

    expon_quantity = create_fourth_poly_model_mnk(np.arange(0, len(time_row_quantity)), time_row_quantity, 'МНК за кількістю проданого товару - екстраполяція', 0.5)
    plot_three_charts(expon_quantity, mnk_quantity, time_row_quantity, 'Передбачення', 'Тренд', 'Вхідні дані', 'Екстраполяція за МНК кількості проданого товару')
    expon_revenue = create_fourth_poly_model_mnk(np.arange(0, len(mnk_revenue)), mnk_revenue, 'МНК за виручкою - екстраполяція', 0.5)
    plot_three_charts(expon_revenue, mnk_revenue, time_row_revenue, 'Передбачення', 'Тренд', 'Вхідні дані', 'Екстраполяція за МНК виручки')
    expon_quantity_by_months = create_fourth_poly_model_mnk(np.arange(0, len(mnk_quantity_by_months)), mnk_quantity_by_months, 'МНК за кількістю проданого товару по місяцям - екстраполяція', 0.5)
    plot_three_charts(expon_quantity_by_months, mnk_quantity_by_months, all_months_quantity, 'Передбачення', 'Тренд', 'Вхідні дані', 'Екстраполяція за МНК кількості проданого товару по місяцям')
    expon_revenue_by_month = create_fourth_poly_model_mnk(np.arange(0, len(mnk_revenue_by_months)), mnk_revenue_by_months, 'МНК за виручкою по місяцям - екстраполяція', 0.5)
    plot_three_charts(expon_revenue_by_month, mnk_revenue_by_months, all_months_revenue, 'Передбачення', 'Тренд', 'Вхідні дані', 'Екстраполяція за МНК виручки по місяцям')


def create_fourth_poly_model_mnk(x, y, title, x_extrapolate_coefficient=None):
    mnk_coeficients = np.polyfit(x, y, 4)
    print(f'Функція моделі ({title}):')
    print(f'y = {mnk_coeficients[0]:.3f}*x^4 + {mnk_coeficients[1]:.3f}*x^3 + {mnk_coeficients[2]:.3f}*x^2 +'
          f' {mnk_coeficients[3]:.3f}*x + {mnk_coeficients[4]:.3f}')
    if x_extrapolate_coefficient is None:
        y_model = [mnk_coeficients[0] * i ** 4 + mnk_coeficients[1] * i ** 3 + mnk_coeficients[2] * i ** 2 +
                   mnk_coeficients[3] * i + mnk_coeficients[4] for i in x]
        plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title=title)
        return y_model
    else:
        y_model = [mnk_coeficients[0] * i ** 4 + mnk_coeficients[1] * i ** 3 + mnk_coeficients[2] * i ** 2 +
                   mnk_coeficients[3] * i + mnk_coeficients[4] for i in
                   range(int(len(y) + len(y) * x_extrapolate_coefficient))]
        return y_model


def stat_characteristics(y_real, y_model, title, coef_of_det=False):
    slo = np.zeros(len(y_real))
    for i in range(len(y_real)):
        slo[i] = y_real[i] - y_model[i]
    mS = np.mean(slo)
    dS = np.var(slo)
    scvS = np.sqrt(dS)
    print(f'Статистичні характеристики {title}:')
    print(f'Математичне сподівання: {mS}')
    print(f'Дисперсія: {dS}')
    print(f'Середньоквадратичне відхилення: {scvS}')
    if coef_of_det:
        coefficient_of_deternimation = 1
        top, bottom = 0, 0
        y_real_average = sum(y_real) / len(y_real)
        for i in range(len(y_real)):
            top += (y_real[i] - y_model[i]) ** 2
            bottom += (y_real[i] - y_real_average) ** 2
        coefficient_of_deternimation -= top / bottom
        print(f'Достовірність апроксимації: {coefficient_of_deternimation}')
    print('------------------------------')


def plot_two_charts(x_values, y1_values, y2_values, y1_label, y2_label, title='', xlabel='', ylabel=''):
    plt.clf()
    plt.plot(x_values, y1_values, label=y1_label)
    plt.plot(x_values, y2_values, label=y2_label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()
    pass


def plot_three_charts(y1_values, y2_values, y3_values, y1_label, y2_label, y3_label, title='', xlabel='', ylabel=''):
    plt.clf()
    plt.plot(np.arange(len(y1_values)), y1_values, label=y1_label)
    plt.plot(np.arange(len(y2_values)), y2_values, label=y2_label)
    plt.plot(np.arange(len(y3_values)), y3_values, label=y3_label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    main()

