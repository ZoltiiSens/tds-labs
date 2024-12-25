"""
Виконав: Литвиненко Роман
Лабораторна робота №1, III рівень складності
Вибірка даних - ціна BTC у проміжку часу 01.07.2023-01.07.2024
"""


import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import matplotlib
matplotlib.use('TkAgg')


# Константи для генерування шуму
LINEAR_A = -7000
LINEAR_B = 7000
NORMAL_M = 0
NORMAL_D = 4000
EXPONENTIAL_L = 3000
CHISQUARE_K = 1
ABNORMAL_MISTAKES_NUMBER = 100


# Мейн функція
def main():
    # Парсинг даних
    user_answer = input('Оберіть, звідки парсити дані:\n * 0 - Сайт\n * 1 - Файл\nEnter: ')
    if user_answer == '0':
        input_data = parse_data_from_coinbase()
    elif user_answer == '1':
        input_data = read_data_from_xlsx('BTC_price_01_07_2023__01_07_2024.xlsx')
    else:
        print('Помилка при введені!')
        return
    print('---------- Парсинг завершено ----------')

    # Отримання необхідних даних з парсингу
    y_values_real = [float(i[1]) for i in input_data]
    x_values_real = [i for i in range(len(y_values_real))]
    DATA_SIZE = len(y_values_real)
    plot_chart(y_values_real, title='Ціна біткоїну 01.07.2023-01.07.2023', ylabel='Ціна BTC, $', xlabel='Час, дні')

    # Вибір типу моделі, отримання коефіцієнтів, її побудова
    user_answer = input('Оберіть тип моделі(за МНК):\n * 0 - лінійна\n * 1 - квадратична\n * 2 - кубічна\n * 3 - 4 '
                        'степінь\nEnter: ')
    if user_answer == '0':
        y_values_model = create_linear_model_mnk(x_values_real, y_values_real)
    elif user_answer == '1':
        y_values_model = create_quadratic_model_mnk(x_values_real, y_values_real)
    elif user_answer == '2':
        y_values_model = create_cubic_model_mnk(x_values_real, y_values_real)
    elif user_answer == '3':
        y_values_model = create_fourth_poly_model_mnk(x_values_real, y_values_real)
    else:
        print('Помилка при введені!')
        return
    print('------------------------------')

    # Отримання статистичних характеристик в залежності від моделі
    stat_characteristics(y_values_real, y_values_model, 'РЕАЛЬНИХ ДАНИХ', coef_of_det=True)

    # Вивід гістограми реальних даних в залежностіі від тренду
    slo = np.zeros(len(y_values_real))
    for i in range(len(y_values_real)):
        slo[i] = y_values_real[i] - y_values_model[i]
    plt.hist(slo, bins=20)
    plt.show()

    # Вибір закону розподілу для генерування шуму для моделі, її генерація, вивід статистичних карактеристик
    user_answer = input('Оберіть тип закону зміни похибки:\n * 0 - рівномірний\n * 1 - нормальний\n * 2 - '
                        'експоненційний\n * 3 - хі-квадрат\nEnter: ')
    if user_answer == '0':
        y_mistake_values = generate_linear_mistake(DATA_SIZE)
    elif user_answer == '1':
        y_mistake_values = generate_normal_mistake(DATA_SIZE)
    elif user_answer == '2':
        y_mistake_values = generate_exponential_mistake(DATA_SIZE)
    elif user_answer == '3':
        y_mistake_values = generate_hi_squared_mistake(DATA_SIZE)
    else:
        print('Помилка при введені!')
        return

    # Побудова моделі з шумом, вивід її статистичних характеристик, побудова графіку
    y_values_with_mistake = np.zeros(DATA_SIZE)
    for i in range(DATA_SIZE):
        y_values_with_mistake[i] = y_values_model[i] + y_mistake_values[i]
    stat_characteristics(y_values_with_mistake, y_values_model, 'МОДЕЛІ З ШУМОМ')
    plot_two_charts(x_values_real, y_values_with_mistake, y_values_model, 'Модель + шум', 'Модель',
                    title='Модель з шумом')

    # Побудова моделі з шумом та аномальними помилками, вивід її статистичних характеристик, побудова графіку
    y_abnormal_values = generate_abnormal_mistakes__normal(DATA_SIZE)
    y_values_with_mistake_and_abnormal = y_values_with_mistake
    # noinspection PyTypeChecker
    for i in range(DATA_SIZE):
        y_values_with_mistake_and_abnormal[i] += y_abnormal_values[i]
    stat_characteristics(y_values_with_mistake_and_abnormal, y_values_model, 'МОДЕЛІ З ШУМОМ ТА АНОМАЛЬНИМИ ПОМИЛКАМИ')
    plot_two_charts(x_values_real, y_values_with_mistake_and_abnormal, y_values_model,
                    'Модель + шум + аномальні помилки', 'Модель', title='Модель з шумом та аномальними помилками')


# Функції парсингу ціни біткоїну
def parse_data_from_coinbase(url='https://coinmarketcap.com/currencies/bitcoin/historical-data/'):
    """
    Function parses bitcoin price (daily, close price) from 1st July 2023 to 1st July 2024
    :param url: URL from which we parse BTC price data, default - coinmarketcap.com
    :return: list [ list[] ]: list with dates and prices of BTC
    """

    # Створюємо інстанс вебдрайверу, відкриваємо посилання
    print(f'Парсинг з сайту {url}...')
    chrome_webdriver = webdriver.Chrome()
    chrome_webdriver.get(url)
    time.sleep(2)

    # Встановлюємо необхідні дати (З 1 липня 2023 до 1 липня 2024)
    chrome_webdriver.find_element(By.CSS_SELECTOR, '.BaseButton_size-md__9TpuT').click()
    button_left = chrome_webdriver.find_element(By.CSS_SELECTOR, '.icon-Chevron-left')
    button_right = chrome_webdriver.find_element(By.CSS_SELECTOR, '.icon-Chevron-right')
    for i in range(12):
        button_left.click()
        time.sleep(0.1)
    chrome_webdriver.find_element(By.CSS_SELECTOR, '[aria-label="Choose Saturday, July 1st, 2023"]').click()
    for i in range(12):
        button_right.click()
        time.sleep(0.1)
    chrome_webdriver.find_element(By.CSS_SELECTOR, '[aria-label="Choose Monday, July 1st, 2024"]').click()
    chrome_webdriver.find_element(By.CSS_SELECTOR, '.iVwhOF').click()
    time.sleep(2)

    # Скролимо вниз сторінки, щоб завантажити всю таблицю
    chrome_webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    # Обробляємо отрману інформацію зі сторінки - отримуємо список зі списками "дата - ціна закриття торгів"
    data = BeautifulSoup(chrome_webdriver.page_source, features="html.parser")
    table_rows = data.find('tbody').findAll('tr')
    result = []
    return_result = []
    for table_row in table_rows:
        tds = table_row.findAll('td')
        price = tds[4].contents[0].replace('$', '').replace(',', '')
        result.append([tds[0].contents[0], price])
        return_result.append(price)

    # Зберігаємо дані у файл exel
    df = pd.DataFrame(result, columns=["Date", "Close price"])
    df.to_excel("BTC_price_01_07_2023__01_07_2024.xlsx", index=False)

    # Повертаємо дані у вигляді списку
    return result


def read_data_from_xlsx(path):
    print(f'Парсинг з файлу {path}...')
    dates = pd.read_excel(path)['Date'].tolist()
    close_prices = pd.read_excel(path)['Close price'].tolist()
    result = []
    for i in range(len(dates)):
        result.append([dates[i], close_prices[i]])
    return result


# Функції для генерації моделі за методом найменших квадратів (поліноми 1-4 ступенів)
def create_linear_model_mnk(x, y):
    mnk_coeficients = np.polyfit(x, y, 1)
    y_model = [mnk_coeficients[0] * i + mnk_coeficients[1] for i in x]
    print('Функція моделі:')
    print(f'y = {mnk_coeficients[0]:.3f}*x + {mnk_coeficients[1]:.3f}')
    plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title='Лінійна модель(МНК)', xlabel='Час, дні',
                    ylabel='Ціна BTC, $')
    return y_model


def create_quadratic_model_mnk(x, y):
    mnk_coeficients = np.polyfit(x, y, 2)
    y_model = [mnk_coeficients[0] * i ** 2 + mnk_coeficients[1] * i + mnk_coeficients[2] for i in x]
    print('Функція моделі:')
    print(f'y = {mnk_coeficients[0]:.3f}*x^2 + {mnk_coeficients[1]:.3f}*x + {mnk_coeficients[2]:.3f}')
    plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title='Квадратична модель(МНК)', xlabel='Час, дні',
                    ylabel='Ціна BTC, $')
    return y_model


def create_cubic_model_mnk(x, y):
    mnk_coeficients = np.polyfit(x, y, 3)
    y_model = [mnk_coeficients[0] * i ** 3 + mnk_coeficients[1] * i ** 2 + mnk_coeficients[2] * i + mnk_coeficients[3]
               for i in x]
    print('Функція моделі:')
    print(f'y = {mnk_coeficients[0]:.3f}*x^3 + {mnk_coeficients[1]:.3f}*x^2 + {mnk_coeficients[2]:.3f}*x + '
          f'{mnk_coeficients[3]:.3f}')
    plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title='Кубічна модель(МНК)', xlabel='Час, дні',
                    ylabel='Ціна BTC, $')
    return y_model


def create_fourth_poly_model_mnk(x, y):
    mnk_coeficients = np.polyfit(x, y, 4)
    y_model = [mnk_coeficients[0] * i ** 4 + mnk_coeficients[1] * i ** 3 + mnk_coeficients[2] * i ** 2 +
               mnk_coeficients[3] * i + mnk_coeficients[4] for i in x]
    print('Функція моделі:')
    print(f'y = {mnk_coeficients[0]:.3f}*x^4 + {mnk_coeficients[1]:.3f}*x^3 + {mnk_coeficients[2]:.3f}*x^2 + '
          f'{mnk_coeficients[3]:.3f}*x + {mnk_coeficients[4]:.3f}')
    plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title='Поліном 4 степеня(МНК)', xlabel='Час, дні',
                    ylabel='Ціна BTC, $')
    return y_model


# Функція для розрахунку статистичних характеристик вибірки
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


# Функції генерування шуму за законами розподілу
def generate_linear_mistake(size):
    y_values = np.random.uniform(low=LINEAR_A, high=LINEAR_B, size=size)
    plt.hist(y_values, bins=10)
    plt.show()
    return y_values


def generate_normal_mistake(size):
    y_values = np.random.normal(loc=NORMAL_M, scale=NORMAL_D, size=size)
    plt.hist(y_values, bins=20)
    plt.show()
    return y_values


def generate_exponential_mistake(size):
    y_values = np.random.exponential(scale=EXPONENTIAL_L, size=size)
    randoms = np.random.choice([-1, 1], size=len(y_values))
    y_values = y_values * randoms
    plt.hist(y_values, bins=20)
    plt.show()
    return y_values


def generate_hi_squared_mistake(size):
    y_values = np.random.chisquare(df=CHISQUARE_K, size=size)
    y_values = y_values * 20
    plt.hist(y_values, bins=20)
    plt.show()
    return y_values


# Функція генерування аномальних помилок(тільки за нормальним законом, адже роботу інших законів продемонстровано при
# генеруванні шуму)
def generate_abnormal_mistakes__normal(size):
    abnormal_mistakes_result = np.zeros(size)
    abnormal_mistakes = np.random.normal(loc=NORMAL_M, scale=NORMAL_D * 3, size=ABNORMAL_MISTAKES_NUMBER)
    for i in range(ABNORMAL_MISTAKES_NUMBER):
        abnormal_mistakes_result[i] = abnormal_mistakes[i]
    np.random.shuffle(abnormal_mistakes_result)
    return abnormal_mistakes_result


# Функції дял побудови графіків
def plot_chart(data, title='', ylabel='', xlabel=''):
    plt.clf()
    plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


def plot_two_charts(x_values, y1_values, y2_values, y1_label, y2_label, title='', xlabel='', ylabel='',):
    plt.clf()
    plt.plot(x_values, y1_values, label=y1_label)
    plt.plot(x_values, y2_values, label=y2_label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()
    pass


# Вхід у програму
if __name__ == '__main__':
    main()


# def coefficients_1_mnk(x, y):
#     size = len(x)
#     numerator_w1 = size * sum(x[i] * y[i] for i in range(0, size)) - sum(x) * sum(y)
#     denominator = size * sum((x[i]) ** 2 for i in range(0, size)) - (sum(x)) ** 2
#     numerator_w0 = -sum(x) * sum(x[i] * y[i] for i in range(0, size)) +
#                    sum((x[i]) ** 2 for i in range(0, size)) * sum(y)
#     w1 = numerator_w1 / denominator
#     w0 = numerator_w0 / denominator
#     return w1, w0
