"""
Виконав: Литвиненко Роман
Лабораторна робота №2, III рівень складності
Групи вимог №1 та №2, з групи вимог №3 обрано завдання 3.1. Власні алгоритм виявлення аномальних вимірів представлено у
функціях my_AV_cleaning_algorythm1-3
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

# Константи для генерування шуму
NORMAL_M = 0
NORMAL_D = 4000
EXPONENTIAL_L = 3000
ABNORMAL_MISTAKES_NUMBER = 50

# Конастанти для очищення аномальних вимірів
WINDOW_SIZE = 5
Q_MEDIUM = 1.5
Q_MNK_MODIFIED = 0.0015
Q_MY_ALG1 = 1.5
Q_MY_ALG2 = 10

# Константи для екстраполяції
EXTRAPOLATION_MULTIPLIER = 0.5


# Мейн функція
def main():
    # Парсинг даних, отримання вхідних даних
    input_data = read_data_from_xlsx('BTC_price_01_07_2023__01_07_2024.xlsx')
    y_values_real = [float(i[1]) for i in input_data]
    x_values_real = [i for i in range(len(y_values_real))]
    DATA_SIZE = len(y_values_real)
    plot_chart(y_values_real, title='Ціна біткоїну 01.07.2023-01.07.2023', ylabel='Ціна BTC, $', xlabel='Час, дні')

    # Вибір типу моделі, отримання коефіцієнтів, її побудова
    user_answer = input('Оберіть тип моделі(за МНК):\n * 0 - лінійна\n * 1 - квадратична\n * 2 - кубічна\n * 3 - 4 '
                        'степінь\nEnter: ')
    if user_answer == '0':
        y_values_model = create_linear_model_mnk(x_values_real, y_values_real)
        model_degree = 1
    elif user_answer == '1':
        y_values_model = create_quadratic_model_mnk(x_values_real, y_values_real)
        model_degree = 2
    elif user_answer == '2':
        y_values_model = create_cubic_model_mnk(x_values_real, y_values_real)
        model_degree = 3
    elif user_answer == '3':
        y_values_model = create_fourth_poly_model_mnk(x_values_real, y_values_real)
        model_degree = 4
    else:
        print('Помилка при введені!')
        return
    print('------------------------------')

    # Отримання статистичних характеристик в залежності від моделі
    stat_characteristics(y_values_real, y_values_model, 'РЕАЛЬНИХ ДАНИХ', coef_of_det=True)

    # Побудова моделі з шумом та нормальними аномальними вимірами, вивід її статистичних характеристик, побудова графіку
    y_mistake_values = generate_mistake__exponential(DATA_SIZE)
    y_values_with_mistake = np.zeros(DATA_SIZE)
    for i in range(DATA_SIZE):
        y_values_with_mistake[i] = y_values_model[i] + y_mistake_values[i]

    y_abnormal_values = generate_abnormal_mistakes__normal(DATA_SIZE)
    y_model = y_values_with_mistake
    for i in range(DATA_SIZE):
        y_model[i] += y_abnormal_values[i]

    stat_characteristics(y_model, y_values_model, 'МОДЕЛІ З ШУМОМ ТА АНОМАЛЬНИМИ ПОМИЛКАМИ')
    plot_two_charts(x_values_real, y_model, y_values_model, 'Модель + шум(експ) + аномальні виміри(норм)', 'Модель',
                    title='Модель з шумом та аномальними помилками')

    # Вибір способу очищення вхідних даних, очищеня, статистичні характеристики та графік
    user_answer = input('Оберіть спосіб очищення даних від аномальних вимірів:\n * 0 - sliding_wind\n * 1 - medium\n * '
                        '2 - modified_mnk\n3 - my_alg1\n4 - my_alg2\n5 - my_alg3\nEnter: ')
    if user_answer == '0':
        y_model_cleared = sliding_wind(y_model, window_n=WINDOW_SIZE)
    elif user_answer == '1':
        y_model_cleared = medium(y_model, window_n=WINDOW_SIZE, Q=Q_MEDIUM)
    elif user_answer == '2':
        y_model_cleared = mnk_modified(y_model, window_n=WINDOW_SIZE, Q=Q_MNK_MODIFIED, degree=model_degree)
    elif user_answer == '3':
        y_model_cleared = my_AV_cleaning_algorythm1(y_model, window_n=WINDOW_SIZE, Q1=Q_MY_ALG1, Q2=Q_MY_ALG1)
    elif user_answer == '4':
        y_model_cleared = my_AV_cleaning_algorythm2(y_model, Q_MY_ALG2)
    elif user_answer == '5':
        y_model_cleared = my_AV_cleaning_algorythm3(y_model, window_n=WINDOW_SIZE, Q1=Q_MEDIUM, Q2=Q_MNK_MODIFIED,
                                                    Q3=Q_MY_ALG1, degree=model_degree)
    else:
        print('Помилка при введені!')
        return

    # Порівняння вихідних статистичних характеристик для кожного з алгоритмів
    # stat_characteristics(sliding_wind(y_model, WINDOW_SIZE), y_values_model, 'sliding wind')
    # stat_characteristics(medium(y_model, WINDOW_SIZE, Q_MEDIUM), y_values_model, 'medium')
    # stat_characteristics(mnk_modified(y_model, 5, 0.0015, model_degree), y_values_model, 'mnk_modified')
    # stat_characteristics(my_AV_cleaning_algorythm1(y_model, 5, 1.5, 1.5), y_values_model, 'my_AV_cleaning_algorythm1')
    # stat_characteristics(my_AV_cleaning_algorythm2(y_model, 10), y_values_model, 'my_AV_cleaning_algorythm2')
    # stat_characteristics(my_AV_cleaning_algorythm3(y_model, WINDOW_SIZE, Q_MEDIUM, 0.0015, 1.5, model_degree),
    #                      y_values_model, 'my_AV_cleaning_algorythm3')
    # plot_three_charts(y_model_cleared, y_values_model, y_model, 'Модель очищена', 'Модель', 'lalala')
    # plot_two_charts(x_values_real, y_model, y_model_cleared, 'Модель очищена', 'Модель',
    #                 title='Модель з шумом та аномальними помилками')

    stat_characteristics(y_model_cleared, y_values_model, 'ОЧИЩЕНОЇ МОДЕЛІ', coef_of_det=True)
    plot_two_charts(x_values_real, y_model_cleared, y_values_model, 'Модель очищена', 'Модель',
                    title='Модель з шумом та аномальними помилками')

    # Отримання екстрополяційної моделі за МНК, її графік
    if model_degree == 1:
        y_model_extrapolated = create_linear_model_mnk(x_values_real, y_values_real,
                                                       x_extrapolate_coefficient=EXTRAPOLATION_MULTIPLIER)
    elif model_degree == 2:
        y_model_extrapolated = create_quadratic_model_mnk(x_values_real, y_values_real,
                                                          x_extrapolate_coefficient=EXTRAPOLATION_MULTIPLIER)
    elif model_degree == 3:
        y_model_extrapolated = create_cubic_model_mnk(x_values_real, y_values_real,
                                                      x_extrapolate_coefficient=EXTRAPOLATION_MULTIPLIER)
    elif model_degree == 4:
        y_model_extrapolated = create_fourth_poly_model_mnk(x_values_real, y_values_real,
                                                            x_extrapolate_coefficient=EXTRAPOLATION_MULTIPLIER)
    plot_three_charts(y_model_extrapolated, y_model, y_values_model, 'МНК передбачення', 'Модель з шумом та АВ',
                      'Вхідний МНК тренд', 'Екстраполяція за МНК')

    # Вибір фільтру Калмана, фільтрація, побудова графіку та отримання статистичних характеристик
    user_answer = input('Оберіть тип фільтру:\n * 0 - alpha-beta\n * 1 - alpha-beta-gamma\n * 2 - модифікований МНК'
                        '\nEnter: ')
    if user_answer == '0':
        y_model_filtered = alpha_beta(y_model)
        filter_type = 'alpha-beta'
    elif user_answer == '1':
        y_model_filtered = alpha_beta_gamma(y_model)
        filter_type = 'alpha-beta-gamma'
    else:
        print('Помилка при введені!')
        return
    stat_characteristics(y_values_model, y_model_filtered, f'ПРОФІЛЬТРОВАНОЇ МОДЕЛІ {filter_type}', True)
    plot_three_charts(y_model, y_model_filtered, y_values_model, 'Модель з шумом та АВ',
                      f'Результати фільрування {filter_type}', 'Вхідний ідеальний тренд')


# Функції для очищення аномальних значень + 3 власноруч придумані алгоритми
def sliding_wind(y_values, window_n=5):
    result = np.zeros(len(y_values))

    # Прохід ковзного вікна зліва направо
    for i in range(len(y_values) - window_n + 1):
        window = y_values[i:i + window_n]
        mean = np.mean(window)
        result[i + window_n - 1] = mean

    # Зворотний прохід ковзного вікна
    if 2 * window_n <= len(y_values):
        sub_X = y_values[:2 * window_n]
        for i in range(2 * window_n - window_n, -1, -1):
            window = sub_X[i:i + window_n]
            mean = np.mean(window)
            result[i] = mean
    else:
        for i in range(window_n - 1):
            result[i] = y_values[i]

    return result


def medium(y_values, window_n=5, Q=1.5):
    result = y_values.copy()
    window = y_values[:window_n]
    sigma_std = np.std(window)

    for i in range(len(y_values) - window_n + 1):
        window = y_values[i:i + window_n]
        mean_curr = np.mean(window)
        sigma_curr = np.std(window)
        if sigma_curr > Q * sigma_std:
            result[i+window_n-1] = mean_curr

    return result


def mnk_modified(y_values, window_n=5, Q=0.0015, degree=3):
    result = y_values.copy()
    poly_coefficients = np.polyfit(np.arange(len(y_values)), y_values, degree)
    speed_standard = poly_coefficients[1]

    for i in range(len(y_values) - window_n + 1):
        window = y_values[i:i + window_n]
        sigma_curr = np.std(window)
        indicator_1 = abs(speed_standard * np.sqrt(len(y_values)))
        indicator_2 = abs(Q * sigma_curr * speed_standard * np.sqrt(window_n))
        if indicator_2 > indicator_1:
            result[i + window_n - 1] = np.polyval(poly_coefficients, i + window_n - 1)
    return result


def my_AV_cleaning_algorythm1(y_values, window_n=10, Q1=1.5, Q2=1.5):
    """
    Робота цього алгоритму схожа на роботу звичайного sliding_window, проте відрізняється від нього тим, що очищаються
    тільки значення, що є найменшими бо найбільшими на проміжку для уточнення результатів. Це зменншує кількість
    очищених значень, проте уточнює їх. Після основного проходження алгоритму, задля уточнення результатів, відбуваєтсья
    зворотній хід
    """
    input_copy = y_values.copy()
    result = y_values.copy()
    for i in range(len(y_values) - window_n + 1):
        window = input_copy[i:i + window_n]
        mean_curr = np.mean(window)
        sigma_curr = np.std(window)
        if (max(window) == window[-1] and abs(window[-1] - mean_curr) > Q1 * sigma_curr) or\
           (min(window) == window[-1] and abs(window[-1] - mean_curr) > Q1 * sigma_curr):
            result[i + window_n - 1] = mean_curr
    input_copy = result.copy()
    for i in range(len(y_values) - window_n, 0, -1):
        window = input_copy[i:i + window_n]
        mean_curr = np.mean(window)
        sigma_curr = np.std(window)
        if (max(window) == window[-1] and abs(window[-1] - mean_curr) > Q2 * sigma_curr) or\
           (min(window) == window[-1] and abs(window[-1] - mean_curr) > Q2 * sigma_curr):
            result[i] = mean_curr
    return result


def my_AV_cleaning_algorythm2(y_values, Q=10):
    """
    Робота цього алгоритму дещо спрощена, в порівнянні з минулими - проходження по вибірці відбувається повністю, без
    використання принципу sliding_window. Для кожного значення перевіряється його зміна відносно попереднього та, у
    випадку, коли він більше за встановлений критерій(що залежить від значення максимального та мінімального виміру на
    вибірці та вхідного коефіцієнту Q), то результат змінюється на середнє арифметичне його сусідніх вимірів
    """
    criteria = (max(y_values) - min(y_values)) / Q
    result = y_values.copy()

    for i in range(1, len(y_values) - 1):
        if abs(result[i] - result[i-1]) > criteria:
            result[i] = (result[i + 1] + result[i - 1]) / 2
    return result


def my_AV_cleaning_algorythm3(y_values, window_n=5, Q1=1.5, Q2=0.0015, Q3=1.5, degree=3, faithfulness=2):
    """
    Робота цього алгоритму зав'язана на використанні трьох різних способів очищаення аномальних вимірів. Для кожного
    значення даних з вибірки, проводиться перевірка методами(sliding_window, mnk_modified та моїм першим алгоритмом), а
    далі, в залежності від значення параметру faithfulness, перевіряється, скільки з алгоритмів "вважають" значення
    аномальним. У випадку, якщо достатня кількість алгоритмів "згодні" між собою - значення змінної замінюється на
    середнє значення по вікну
    """
    result = y_values.copy()
    window = y_values[:window_n]
    sigma_std = np.std(window)
    speed_standard = np.polyfit(np.arange(len(y_values)), y_values, degree)[1]

    for i in range(len(y_values) - window_n + 1):
        window = y_values[i:i + window_n]
        mean_curr = np.mean(window)
        sigma_curr = np.std(window)
        indicator_1 = abs(speed_standard * np.sqrt(len(y_values)))
        indicator_2 = abs(Q2 * sigma_curr * speed_standard * np.sqrt(window_n))
        toChange = 0
        if sigma_curr > Q1 * sigma_std:
            toChange += 1
            print(True)
        if indicator_2 > indicator_1:
            toChange += 1
            print(True)
        if (max(window) == window[-1] and abs(window[-1] - mean_curr) > Q3 * sigma_curr) or \
           (min(window) == window[-1] and abs(window[-1] - mean_curr) > Q3 * sigma_curr):
            toChange += 1
            print(True)
        if toChange >= faithfulness:
            result[i + window_n - 1] = mean_curr
        print('---')

    return result


# Функції Фльтрів Калмана alfa-beta та alfa-beta-gamma
def alpha_beta(y_values):
    T0 = 1
    result = [0 for _ in range(len(y_values))]
    result[0] = y_values[0]
    speed = (y_values[1] - y_values[0]) / T0
    y_predicted = y_values[0]
    counter = 1
    for i in range(1, len(y_values)):
        alpha = (2 * (2 * counter - 1)) / (counter * (counter + 1))
        beta = 6 / (counter * (counter + 1))
        counter = counter + 1 if counter != 100 else 50     # "Обнулення" пам'яті фільтру задля подолання розбідності

        result[i] = y_predicted + alpha * (y_values[i] - y_predicted)
        speed = speed + beta / T0 * (y_values[i] - y_predicted)
        y_predicted = result[i] + T0 * speed
    return result


def alpha_beta_gamma(y_values):
    T0 = 1
    result = [0 for _ in range(len(y_values))]
    result[0] = y_values[0]
    speed_predicted = (y_values[1] - y_values[0]) / T0
    y_predicted = y_values[0]
    acceleration_predicted = 0
    counter = 1
    for i in range(1, len(y_values)):
        alpha = (3 * (3 * counter ** 2 - 3 * counter + 2)) / (counter * (counter + 1) * (counter + 2))
        beta = (18 * (2 * counter - 1)) / (T0 * (counter + 1) * (counter + 2) * counter)
        gamma = 60 / (T0 ** 2 * (counter + 1) * (counter + 2) * counter)
        counter = counter + 1 if counter != 100 else 50     # "Обнулення" пам'яті фільтру задля подолання розбідності

        result[i] = y_predicted + alpha * (y_values[i] - y_predicted)
        speed = speed_predicted + beta / T0 * (y_values[i] - y_predicted)
        acceleration = acceleration_predicted + gamma / T0 ** 2 * (y_values[i] - y_predicted)
        y_predicted = result[i] + speed * T0 + acceleration * T0 / 2
        speed_predicted = speed + acceleration * T0
        acceleration_predicted = acceleration
    return result


# =================================================== Старі функції ====================================================
# Функція парсингу ціни біткоїну
def read_data_from_xlsx(path):
    print(f'Парсинг з файлу {path}...')
    dates = pd.read_excel(path)['Date'].tolist()
    close_prices = pd.read_excel(path)['Close price'].tolist()
    result = []
    for i in range(len(dates)):
        result.append([dates[i], close_prices[i]])
    return result


# Функції для генерації моделі за методом найменших квадратів (поліноми 1-4 ступенів) + додано можливість побудови
# екстаполяційної моделі
def create_linear_model_mnk(x, y, x_extrapolate_coefficient=None):
    mnk_coeficients = np.polyfit(x, y, 1)
    print('Функція моделі:')
    print(f'y = {mnk_coeficients[0]:.3f}*x + {mnk_coeficients[1]:.3f}')
    if x_extrapolate_coefficient is None:
        y_model = [mnk_coeficients[0] * i + mnk_coeficients[1] for i in x]
        plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title='Лінійна модель(МНК)', xlabel='Час, дні',
                        ylabel='Ціна BTC, $')
        return y_model
    else:
        y_model = [mnk_coeficients[0] * i + mnk_coeficients[1] for i in range(int(len(y) + len(y) *
                                                                                  x_extrapolate_coefficient))]
        return y_model


def create_quadratic_model_mnk(x, y, x_extrapolate_coefficient=None):
    mnk_coeficients = np.polyfit(x, y, 2)
    print('Функція моделі:')
    print(f'y = {mnk_coeficients[0]:.3f}*x^2 + {mnk_coeficients[1]:.3f}*x + {mnk_coeficients[2]:.3f}')
    if x_extrapolate_coefficient is None:
        y_model = [mnk_coeficients[0] * i ** 2 + mnk_coeficients[1] * i + mnk_coeficients[2] for i in x]
        plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title='Квадратична модель(МНК)', xlabel='Час, дні',
                        ylabel='Ціна BTC, $')
        return y_model
    else:
        y_model = [mnk_coeficients[0] * i ** 2 + mnk_coeficients[1] * i + mnk_coeficients[2] for i in
                   range(int(len(y) + len(y) * x_extrapolate_coefficient))]
        return y_model


def create_cubic_model_mnk(x, y, x_extrapolate_coefficient=None):
    mnk_coeficients = np.polyfit(x, y, 3)
    print('Функція моделі:')
    print(f'y = {mnk_coeficients[0]:.3f}*x^3 + {mnk_coeficients[1]:.3f}*x^2 + {mnk_coeficients[2]:.3f}*x +'
          f' {mnk_coeficients[3]:.3f}')
    if x_extrapolate_coefficient is None:
        y_model = [mnk_coeficients[0] * i ** 3 + mnk_coeficients[1] * i ** 2 + mnk_coeficients[2] * i +
                   mnk_coeficients[3] for i in x]
        plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title='Кубічна модель(МНК)', xlabel='Час, дні',
                        ylabel='Ціна BTC, $')
        return y_model
    else:
        y_model = [mnk_coeficients[0] * i ** 3 + mnk_coeficients[1] * i ** 2 + mnk_coeficients[2] * i +
                   mnk_coeficients[3] for i in range(int(len(y) + len(y) * x_extrapolate_coefficient))]
        return y_model


def create_fourth_poly_model_mnk(x, y, x_extrapolate_coefficient=None):
    mnk_coeficients = np.polyfit(x, y, 4)
    print('Функція моделі:')
    print(f'y = {mnk_coeficients[0]:.3f}*x^4 + {mnk_coeficients[1]:.3f}*x^3 + {mnk_coeficients[2]:.3f}*x^2 +'
          f' {mnk_coeficients[3]:.3f}*x + {mnk_coeficients[4]:.3f}')
    if x_extrapolate_coefficient is None:
        y_model = [mnk_coeficients[0] * i ** 4 + mnk_coeficients[1] * i ** 3 + mnk_coeficients[2] * i ** 2 +
                   mnk_coeficients[3] * i + mnk_coeficients[4] for i in x]
        plot_two_charts(x, y, y_model, 'Реальні значення', 'Модель', title='Поліном 4 степеня(МНК)', xlabel='Час, дні',
                        ylabel='Ціна BTC, $')
        return y_model
    else:
        y_model = [mnk_coeficients[0] * i ** 4 + mnk_coeficients[1] * i ** 3 + mnk_coeficients[2] * i ** 2 +
                   mnk_coeficients[3] * i + mnk_coeficients[4] for i in
                   range(int(len(y) + len(y) * x_extrapolate_coefficient))]
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


# Функція генерування шуму за експоненційним законом розподілу
def generate_mistake__exponential(size):
    y_values = np.random.exponential(scale=EXPONENTIAL_L, size=size)
    randoms = np.random.choice([-1, 1], size=len(y_values))
    y_values = y_values * randoms
    return y_values


# Функція генерування аномальних помилок
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


# Вхід у програму
if __name__ == '__main__':
    main()
