"""
Виконав: Литвиненко Роман
Лабораторна робота №1, III рівень складності
Вибірка даних - ціна BTC у проміжку часу 01.07.2023-01.07.2024
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

# Константи для генерування шуму
LINEAR_A = -7000
LINEAR_B = 7000
NORMAL_M = 0
NORMAL_D = 4000
EXPONENTIAL_L = 3000
CHISQUARE_K = 1
ABNORMAL_MISTAKES_NUMBER = 50

WINDOW_SIZE = 5
Q_MEDIUM = 1.5
Q_MNK_MODIFIED = 0.0015


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

    # Генерування шуму за експоненційним законом для моделі
    y_mistake_values = generate_mistake__exponential(DATA_SIZE)
    y_values_with_mistake = np.zeros(DATA_SIZE)
    for i in range(DATA_SIZE):
        y_values_with_mistake[i] = y_values_model[i] + y_mistake_values[i]
    stat_characteristics(y_values_with_mistake, y_values_model, 'МОДЕЛІ З ШУМОМ')

    # Побудова моделі з шумом та нормальними аномальними вимірами, вивід її статистичних характеристик, побудова графіку
    y_abnormal_values = generate_abnormal_mistakes__normal(DATA_SIZE)
    y_model = y_values_with_mistake
    for i in range(DATA_SIZE):
        y_model[i] += y_abnormal_values[i]
    stat_characteristics(y_model, y_values_model, 'МОДЕЛІ З ШУМОМ ТА АНОМАЛЬНИМИ ПОМИЛКАМИ')
    plot_two_charts(x_values_real, y_model, y_values_model,
                    'Модель + шум(експ) + аномальні виміри(норм)', 'Модель',
                    title='Модель з шумом та аномальними помилками')

    # Вибір способу очищення вхідних даних
    user_answer = input('Оберіть спосіб очищення даних від аномальних вимірів:\n * 0 - sliding_wind\n * 1 - medium\n * '
                        '2 - модифікований МНК\nEnter: ')
    if user_answer == '0':
        y_model_cleared = sliding_wind(y_model, WINDOW_SIZE)
    elif user_answer == '1':
        y_model_cleared = medium(y_model, WINDOW_SIZE, Q_MEDIUM)
    elif user_answer == '2':
        y_model_cleared = mnk_modified(y_model, 5, 0.0015, 3)
        pass

    stat_characteristics(y_model_cleared, y_values_model, 'Очищена модель', coef_of_det=True)
    plot_two_charts(x_values_real, y_model_cleared, y_values_model, 'Модель очищена', 'Модель', title='Модель з шумом та аномальними помилками')


# def sliding_wind(y_values, window_n):
#     window = np.zeros(window_n)
#     window_start = 0
#     window_end = window_n
#     result = np.zeros(len(y_values))
#     for i in range(window_n - 1):
#         result[i] = y_values[i]
#     for i in range(len(y_values) - window_n + 1):
#         for j, k in enumerate(range(window_start, window_end)):
#             window[j] = y_values[k]
#         print(window)
#         result[window_n + i - 1] = np.mean(window)
#         window_start += 1
#         window_end += 1
#     return result


def sliding_wind(y_values, window_n):
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


def medium(y_values, window_n, Q):
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


# def clear_anomalies(X, poly_degree=2, Nwin=10, Q=1.5):
#     n = len(X)
#     X_new = np.copy(X)
#
#     # 2. Fit a polynomial model using Least Squares Method (LSM)
#     x_vals = np.arange(n)
#     speed_etalon = np.polyfit(x_vals, X, poly_degree)[1]
#     # print(poly_model)
#     # speed_etalon = poly_model.deriv(2)[2]  # c2, second derivative
#
#
#     #     speed_standard = poly_coefficients[1]
#
#     print(speed_etalon)
#
#     # 3. Sliding window configuration
#     for j in range(Nwin, n - Nwin):
#         # 4. Compute sliding window statistics
#         window = X[j - Nwin:j + Nwin]
#         x_mean_j = np.mean(window)
#         D_j = np.var(window)
#         sigm_j = np.std(window)
#
#         # 5. Compute controlled parameters for anomaly detection
#         ind_1 = np.abs(speed_etalon) * np.sqrt(n)
#         ind_2 = np.abs(Q * sigm_j * speed_etalon * np.sqrt(Nwin))
#
#         # 6. Anomaly detection condition
#         if ind_2 > ind_1:
#             # Replace the j-th value in X_new with the LSM score
#             X_new[j] = x_mean_j
#
#     # 7. Return the dataset cleared of anomalies
#     return X_new
#
#
# def poly_evaluation(coeffs, x_value):
#     poly = np.polynomial.Polynomial(coeffs)
#     return poly(x_value)



# print(mnk_modified([1,2,3,4,5,6,7,8,9,10,11, 10], 4, 7, 3))



# def mnk_polynomial_fit(X, degree=2):
#     """
#     Виконує МНК з поліноміальною моделлю для вибірки X.
#     degree: Степінь поліному (за замовчуванням - квадратична модель).
#     """
#     n = len(X)
#     x_values = np.arange(n)  # Припускаємо, що x = 0, 1, 2, ..., n-1
#     # Поліноміальна регресія
#     p = np.polynomial.Polynomial.fit(x_values, X, degree).convert()  # Отримуємо коефіцієнти полінома
#     return p
#
#
# def Sliding_Window_AV_Detect_MNK(y_values, Nwin, Q):
#     X_cleaned = y_values.copy()
#     coefficients = mnk_polynomial_fit(y_values, degree=2)
#     speed_standard = coefficients.coef[1]  # Коефіцієнт c1 відповідає за швидкість
#
#     # Прохід ковзного вікна по всій вибірці
#     for i in range(Nwin, len(y_values)):
#         # Формування ковзного вікна
#         window = y_values[i - Nwin:i]
#
#         # Поточні оцінки: середнє, дисперсія та sigma
#         mean_curr = np.mean(window)
#         var_curr = np.var(window)
#         sigma_curr = np.sqrt(var_curr)
#
#         # Обчислення контрольованих параметрів аномальності
#         indicator_1 = abs(speed_standard * np.sqrt(len(y_values)))
#         indicator_2 = abs(Q * sigma_curr * speed_standard * np.sqrt(Nwin))
#
#         # Виявлення аномалій
#         if indicator_2 > indicator_1:
#             # Якщо виявлено аномалію, замінюємо значення на МНК оцінку
#             X_cleaned[i] = coefficients(i)
#
#     return X_cleaned


# medium([1, 2, 3, 4, 5, 6, 7], 4, np.sqrt(3))



# Функція парсингу ціни біткоїну
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


# Функція генерування шуму за експоненційним законом розподілу
def generate_mistake__exponential(size):
    y_values = np.random.exponential(scale=EXPONENTIAL_L, size=size)
    randoms = np.random.choice([-1, 1], size=len(y_values))
    y_values = y_values * randoms
    return y_values


# Функція генерування аномальних помилок
def generate_abnormal_mistakes__normal(size):
    abnormal_mistakes_result = np.zeros(size)
    abnormal_mistakes = np.random.normal(loc=NORMAL_M, scale=NORMAL_D * 1.6, size=ABNORMAL_MISTAKES_NUMBER)
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


def plot_two_charts(x_values, y1_values, y2_values, y1_label, y2_label, title='', xlabel='', ylabel='', ):
    plt.clf()
    plt.plot(x_values, y1_values, label=y1_label)
    plt.plot(x_values, y2_values, label=y2_label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()
    pass



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GOOD ONE
# import math as mt
#
# # Polynomial fitting using Least Squares Method (LSM)
# def MNK(S0):
#     iter = len(S0)
#     Yin = np.zeros((iter, 1))
#     F = np.ones((iter, 3))
#
#     for i in range(iter):
#         Yin[i, 0] = float(S0[i])  # Input data
#         F[i, 1] = float(i)  # Linear term
#         F[i, 2] = float(i * i)  # Quadratic term
#
#     # LSM polynomial fitting
#     FT = F.T
#     FFT = FT.dot(F)
#     FFTI = np.linalg.inv(FFT)
#     FFTIFT = FFTI.dot(FT)
#     C = FFTIFT.dot(Yin)
#     Yout = F.dot(C)
#
#     return Yout
#
#
# # Sliding window anomaly detection with LSM
# def Sliding_Window_AV_Detect_MNK(S0, Q, n_Wind):
#     copied_S0 = S0.copy()
#     iter = len(copied_S0)
#     j_Wind = mt.ceil(iter - n_Wind) + 1
#     S0_Wind = np.zeros((n_Wind))
#
#     # Fit polynomial model and get standard speed (2nd derivative)
#     Speed_standart = MNK_AV_Detect(copied_S0)
#     Yout_S0 = MNK(copied_S0)
#
#     for j in range(j_Wind):
#         # Sliding window
#         for i in range(n_Wind):
#             l = j + i
#             S0_Wind[i] = copied_S0[l]
#
#         # Statistical characteristics of the sliding window
#         dS = np.var(S0_Wind)
#         scvS = mt.sqrt(dS)
#
#         # Anomaly detection condition
#         Speed_standart_1 = abs(Speed_standart * mt.sqrt(iter))
#         Speed_1 = abs(Q * Speed_standart * mt.sqrt(n_Wind) * scvS)
#
#         if Speed_1 > Speed_standart_1:
#             # Replace anomaly with polynomial approximation
#             copied_S0[l] = Yout_S0[l, 0]
#
#     return copied_S0
# #
# #
# # Detect polynomial coefficients for anomaly detection (speed model)
# def MNK_AV_Detect(S0):
#     iter = len(S0)
#     Yin = np.zeros((iter, 1))
#     F = np.ones((iter, 3))
#
#     for i in range(iter):
#         Yin[i, 0] = float(S0[i])
#         F[i, 1] = float(i)
#         F[i, 2] = float(i * i)
#
#     # LSM fitting
#     FT = F.T
#     FFT = FT.dot(F)
#     FFTI = np.linalg.inv(FFT)
#     FFTIFT = FFTI.dot(FT)
#     C = FFTIFT.dot(Yin)
#
#     return C[1, 0]  # Return speed (c1 coefficient)


# def MNK(S0, degree):
#     n = len(S0)
#     x = np.arange(n)
#
#     # Construct matrix for polynomial fit: [1, x, x^2, ..., x^degree]
#     F = np.vstack([x ** d for d in range(degree + 1)]).T
#
#     # Solve the normal equation for least squares: (F^T F)C = F^T S0
#     C = np.linalg.lstsq(F, S0, rcond=None)[0]
#
#     # Calculate the fitted values
#     Yout = F.dot(C)
#     return Yout
#
#
# # Sliding window anomaly detection with LSM using variable degree
# def Sliding_Window_AV_Detect_MNK(S0, Q, n_Wind, degree):
#     S0_copy = S0.copy()
#     n = len(S0_copy)
#
#     # Get polynomial fit for the whole dataset
#     Yout_S0 = MNK(S0_copy, degree)
#
#     # Detect the polynomial speed (linear term from LSM fit)
#     Speed_standart = MNK_AV_Detect(S0_copy, degree)
#
#     for j in range(n - n_Wind + 1):
#         # Extract the sliding window
#         S0_Wind = S0_copy[j:j + n_Wind]
#
#         # Compute standard deviation in the window
#         scvS = np.std(S0_Wind)
#
#         # Calculate anomaly detection thresholds
#         Speed_standart_1 = np.abs(Speed_standart * np.sqrt(n))
#         Speed_1 = np.abs(Q * Speed_standart * np.sqrt(n_Wind) * scvS)
#
#         # If the window contains an anomaly, replace it with LSM prediction
#         if Speed_1 > Speed_standart_1:
#             S0_copy[j:j + n_Wind] = Yout_S0[j:j + n_Wind]
#
#     return S0_copy
#
#
# # Detect the linear term (speed) from the LSM fit, generalized for variable degree
# def MNK_AV_Detect(S0, degree):
#     n = len(S0)
#     x = np.arange(n)
#
#     # Construct matrix for polynomial fit: [1, x, x^2, ..., x^degree]
#     F = np.vstack([x ** d for d in range(degree + 1)]).T
#
#     # Solve for coefficients using least squares
#     C = np.linalg.lstsq(F, S0, rcond=None)[0]
#
#     # Return the linear coefficient (speed term)
#     # In general polynomial, the linear term corresponds to C[1] if degree >= 1
#     return C[1] if degree >= 1 else 0
#
#
#
#
#
#
# # arr = np.random.normal(10000, 50, 365)
# arr = [1,2,3,4,5,100,7,8,9,10,11,12]
# new_arr = Sliding_Window_AV_Detect_MNK(arr, 7, 4, 1)
# print(arr, new_arr)
# plot_two_charts(np.arange(len(arr)), arr, new_arr, 'arr', 'new_arr')
# print(dict(zip(arr, new_arr)))
#
#
# test_str = '''[61288.20634755 64510.15236052 62195.97579225 69932.40739482
#  70852.42048199 62428.99514288 69211.23494281 62947.4127453
#  65205.85459849 67126.11362155 52328.99273458 65151.27753168
#  61424.54978796 54996.63373908 69600.0820475  59785.08342612
#  65919.3096029  66363.6938435  65277.59568508 68189.14144821
#  67970.20171678 65920.53117509 71914.5527341  66350.68581769
#  66681.5962318  65851.25664754 65113.91090946 58253.75249877
#  76720.57027492 56989.12367932 69154.47794426 66825.34474742
#  65381.47961272 63759.53922218 58975.92369764 72902.46931747
#  72497.97134317 62894.26574122 62747.90493878 62951.99307445
#  66873.08998733 67643.4649575  70306.94402811 62707.49384653
#  73769.37929999 67403.15078231 63018.39532944 72989.2466467
#  61489.27521458 59027.57151932 62106.32653734 63510.81891098
#  70380.42081225 62342.24712848 67602.99282883 66581.36814638
#  65661.59261251 62803.46116802 65821.86331353 72053.60938622
#  71592.77474034 69552.20701854 58965.87070429 61260.93796332
#  67359.59680922 66363.61734578 71011.26151915 65969.13652871
#  66334.96345182 72578.35464772 69877.97617193 68466.10987673
#  65351.38736972 61088.07458565 69849.05400386 63546.7123686
#  62603.58775346 70016.48243305 59949.92001573 75763.51629786
#  72020.55637112 65449.37962449 64086.51359802 62824.90260637
#  62547.02283239 60360.62866887 63941.72137617 72130.24519084
#  65880.31444056 64509.15354963 52952.67157502 55253.90955918
#  60080.76001801 65544.91350853 62262.89263723 63097.23890012
#  62959.14225443 63526.90828082 62289.34581955 67730.48176822
#  64012.15564944 62724.89829366 57498.29987521 61206.70470927
#  64473.45407499 70660.8793458  60997.15642663 57819.86505662
#  49208.61829425 58605.45612062 60679.23591514 60823.1220222
#  68306.14000169 58068.56501041 52480.54585827 67626.50443844
#  61679.52752567 59487.99224693 68419.50499538 58408.4342905
#  65423.47368195 57026.67025758 59700.90361682 55975.52188661
#  58724.41692667 67416.35966403 63812.31585563 63690.77757731
#  53845.42493582 54756.11468838 59499.12132038 57344.67423713
#  59109.73454335 62029.06506863 54141.14609801 58447.45938294
#  56994.88607252 60041.10139828 66240.16522775 55011.71476177
#  50016.77272235 65048.92898281 55838.37765966 50296.08740919
#  49232.47593975 53895.8071204  51960.67556055 49704.8064441
#  37446.0043412  41631.35053158 52889.96927239 50623.12446253
#  52956.05853793 55249.15620978 52109.48657602 44563.74346679
#  50560.01721787 51969.05577805 54067.75560275 51219.67884056
#  44169.41772948 51957.28534285 38989.98449648 46619.42235185
#  61034.37693728 59475.46982743 64623.72564198 44448.43111426
#  47358.88893876 47743.54378666 49328.37697084 50107.35830809
#  48931.82221357 45384.60804015 34662.97223313 44930.54806297
#  46039.08119679 39585.27215015 50047.56016601 50842.26434552
#  34979.50022454 44131.40308944 35379.74515717 39662.2199237
#  43239.44452329 29681.10025659 40623.67929838 44170.28131606
#  43843.11619687 41815.23864747 42698.997335   43221.95625281
#  41501.23823987 45940.01828022 28659.63473424 41976.95839156
#  49617.88418777 47484.74534387 35073.97538546 40483.74082902
#  42416.72149148 39348.45836265 35179.97964377 45138.75090265
#  39454.526969   44143.95996307 36251.35765291 29610.3555882
#  41281.95953932 27411.57568522 36477.70843025 41675.85010869
#  53060.64437655 38092.38790211 48904.7440446  22725.22099385
#  34080.53469906 36464.453007   36120.63198101 43931.77931566
#  40795.92515294 36114.10104668 37312.39718555 36082.77619137
#  32673.4086326  35604.82090107 33387.45029237 25529.46473164
#  30570.04648141 29311.75291943 35333.47612653 33201.09881279
#  33662.87026877 23942.51489714 22912.25008799 36503.40532202
#  31275.68556357 44500.1197332  33269.60260235 18715.88321171
#  44301.98938085 31096.33666332 28444.62474096 32332.55504448
#  25017.16802755 24845.85702745 36809.41516857 29435.92796538
#  31191.6539642  32431.3826363  35968.71404798 29853.46849567
#  27195.0373992  27732.57272008 29126.62961724 35364.51689156
#  36176.18692408 30193.95923488 31742.96434823 39534.79473388
#  24783.84946831 23158.17889631 27957.37555834 29349.53265991
#  18142.54817237 33319.60393857 28722.50750299 28060.41336828
#  17670.57271798 20415.90980139 21632.42927392 32870.1930412
#  24084.2310184  28276.9911992  21406.32662825 33544.62423838
#  26936.59799232 23214.69544904 27487.78656074 28722.37480522
#  28608.43519604 32687.65440296 22131.70156866 31929.85276474
#  22471.11218259 28913.21084898 22419.31411732 34400.07881496
#  26826.30699891 27902.63561366 30909.70637316 32301.82167768
#  37257.34343985 24506.42177572 25973.24477363 33011.08766668
#  24787.02263347 27014.85136111 26255.8112313  22864.04029881
#  30645.72972717 20095.51722631 27475.57630632 30492.40696413
#  26181.95133827 24077.06130541 24116.32230634 38946.69898351
#  27515.76051837 38595.98374992 25984.32769037 25821.32921507
#  28947.43519973 21614.88706012 27429.6332694  31247.40718977
#  42336.12141064 31084.96755089 27956.59941303 30123.00824694
#  22164.21695465 26868.41871672 27554.85610738 30116.85661126
#  25011.79736151 25102.70904826 32933.41427988 26750.82957785
#  21505.89372211 28824.89738001 28087.04666859 26188.23251967
#  22331.01946436 26835.44165683 19293.55930166 25368.27489596
#  32077.17393405 35379.70259236 29363.5832013  26167.04203442
#  29878.63239975 33312.85834696 30155.0478852  20358.20864151
#  26044.74857795 33760.79448508 18502.99408447 33222.50331648
#  26088.92508985 43204.99182768 31281.5312429  37709.62789288
#  23776.62386695 33542.23685627 30623.25835909 35411.46256295
#  30058.15400615 49206.62615087 34578.68265823 42397.66986141
#  32168.72120622 37730.97545953 32817.51490677 26141.65828337
#  38101.51374161 36142.84905364]
# '''






import math


def mnk_modified(y_values, Nwin, Q, degree=3):
    X_cleaned = y_values.copy()
    poly_coefficients = np.polyfit(np.arange(len(y_values)), y_values, degree)
    speed_standard = poly_coefficients[1]

    for i in range(len(y_values) - Nwin + 1):
        window = y_values[i:i + Nwin]

        mean_curr = np.mean(window)
        var_curr = np.var(window)
        sigma_curr = np.std(window)

        indicator_1 = abs(speed_standard * math.sqrt(len(y_values)))
        indicator_2 = abs(Q * sigma_curr * speed_standard * math.sqrt(Nwin))

        if indicator_2 > indicator_1:
            X_cleaned[i + Nwin - 1] = np.polyval(poly_coefficients, i + Nwin - 1)
            # X_cleaned[i] = np.polyval(poly_coefficients, i)
    return X_cleaned





test_model = [64033.20238988559, 64193.28765549958, 64348.830527970895, 64499.857510961374, 64646.395108132856, 64788.46982314716, 64926.108159666146, 65059.33662135163, 65188.18171186544, 65312.66993486943, 65432.82779402542, 65548.68179299525, 65660.25843544076, 65767.58422502378, 65870.68566540614, 65969.58926024968, 66064.32151321623, 66154.90892796763, 66241.37800816572, 66323.75525747231, 66402.06717954927, 66476.3402780584, 66546.60105666156, 66612.87601902056, 66675.19166879727, 66733.5745096535, 66788.05104525108, 66838.64777925187, 66885.39121531768, 66928.30785711034, 66967.4242082917, 67002.76677252361, 67034.36205346788, 67062.23655478634, 67086.41678014085, 67106.92923319322, 67123.8004176053, 67137.05683703891, 67146.7249951559, 67152.8313956181, 67155.40254208734, 67154.46493822546, 67150.04508769429, 67142.16949415568, 67130.86466127144, 67116.15709270342, 67098.07329211345, 67076.63976316336, 67051.88300951499, 67023.82953483018, 66992.50584277076, 66957.93843699855, 66920.15382117541, 66879.17849896316, 66835.03897402364, 66787.76175001868, 66737.37333061012, 66683.90021945979, 66627.36892022952, 66567.80593658115, 66505.23777217652, 66439.69093067746, 66371.1919157458, 66299.76723104338, 66225.44338023204, 66148.2468669736, 66068.2041949299, 65985.34186776279, 65899.68638913408, 65811.26426270562, 65720.10199213924, 65626.22608109677, 65529.66303324006, 65430.439352230926, 65328.58154173122, 65224.11610540276, 65117.0695469074, 65007.46836990694, 64895.33907806326, 64780.70817503816, 64663.60216449349, 64544.04755009108, 64422.07083549277, 64297.698524360385, 64170.95712035577, 64041.87312714075, 63910.47304837717, 63776.78338772686, 63640.83064885164, 63502.64133541337, 63362.24195107386, 63219.65899949497, 63074.91898433852, 62928.04840926634, 62779.07377794028, 62628.02159402215, 62474.91836117381, 62319.79058305709, 62162.664763333814, 62003.567405665824, 61842.52501371494, 61679.564091143024, 61514.71114161189, 61347.99266878338, 61179.435176319326, 61009.06516788156, 60836.909147131926, 60662.993617732245, 60487.345083344364, 60309.99004763011, 60130.955014251325, 59950.266486869834, 59767.95096914748, 59584.0349647461, 59398.5449773275, 59211.50751055356, 59022.949068086076, 58832.89615358692, 58641.37527071788, 58448.41292314083, 58254.03561451758, 58058.26984850998, 57861.14212877986, 57662.678958989054, 57462.906842799406, 57261.85228387272, 57059.54178587087, 56856.001852455665, 56651.25898728895, 56445.33969403255, 56238.27047634831, 56030.07783789806, 55820.78828234364, 55610.42831334687, 55399.0244345696, 55186.603149673654, 54973.19096232088, 54758.8143761731, 54543.49989489216, 54327.27402213987, 54110.1632615781, 53892.194116868646, 53673.39309167338, 53453.78668965411, 53233.4014144727, 53012.26376979094, 52790.4002592707, 52567.8373865738, 52344.60165536207, 52120.71956929738, 51896.21763204152, 51671.12234725634, 51445.46021860369, 51219.257749745375, 50992.54144434325, 50765.33780605916, 50537.67333855491, 50309.574545492345, 50081.06793053332, 49852.17999733964, 49622.937249573166, 49393.366190895715, 49163.49332496914, 48933.345155455245, 48702.94818601588, 48472.32892031289, 48241.5138620081, 48010.529514763344, 47779.402382240456, 47548.15896810128, 47316.82577600764, 47085.42930962138, 46853.99607260432, 46622.552568618295, 46391.12530132516, 46159.74077438672, 45928.42549146485, 45697.205956221354, 45466.10867231808, 45235.16014341685, 45004.386873179494, 44773.815365267874, 44543.47212334381, 44313.383651069125, 44083.57645210568, 43854.077030115266, 43624.91188875976, 43396.107531700996, 43167.690462600774, 42939.68718512094, 42712.12420292337, 42485.028019669844, 42258.42513902221, 42032.342064642326, 41806.80530019202, 41581.8413493331, 41357.476715727425, 41133.73790303682, 40910.65141492312, 40688.24375504817, 40466.54142707379, 40245.57093466183, 40025.35878147411, 39805.93147117247, 39587.31550741876, 39369.53739387479, 39152.62363420239, 38936.60073206344, 38721.495191119706, 38507.3335150331, 38294.1422074654, 38081.94777207844, 37870.77671253409, 37660.65553249416, 37451.610735620496, 37243.668825574925, 37036.8563060193, 36831.19968061543, 36626.72545302514, 36423.46012691029, 36221.43020593273, 36020.66219375427, 35821.182594036734, 35623.01791044197, 35426.19464663182, 35230.73930626812, 35036.678393012684, 34844.03841052736, 34652.845862473994, 34463.127252514394, 34274.9090843104, 34088.21786152388, 33903.08008781662, 33719.5222668505, 33537.570902287334, 33357.25249778894, 33178.593557017164, 33001.62058363384, 32826.36008130083, 32652.83855367994, 32481.082504433005, 32311.11843722187, 32142.972855708358, 31976.672263554312, 31812.24316442157, 31649.712061971957, 31489.1054598673, 31330.449861769484, 31173.771771340274, 31019.09769224155, 30866.454128135098, 30715.86758268284, 30567.36455954652, 30420.97156238802, 30276.715094869163, 30134.621660651806, 29994.717763397734, 29857.02990676881, 29721.584594426888, 29588.40833003378, 29457.527617251326, 29328.968959741353, 29202.758861165697, 29078.92382518619, 28957.490355464695, 28838.484955663, 28721.934129442998, 28607.864380466483, 28496.302212395283, 28387.27412889125, 28280.8066336162, 28176.926230232, 28075.65942240046, 27977.032713783417, 27881.072608042734, 27787.8056088402, 27697.2582198377, 27609.456944697013, 27524.428287080023, 27442.198750648502, 27362.794839064372, 27286.2430559894, 27212.569905085424, 27141.801890014314, 27073.96551443787, 27009.087282017987, 26947.193696416405, 26888.311261295035, 26832.466480315692, 26779.685857140168, 26729.995895430387, 26683.423098848092, 26639.993971055163, 26599.735015713413, 26562.672736484725, 26528.833637030868, 26498.244221013723, 26470.93099209512, 26446.92045393688, 26426.239110200826, 26408.91346454883, 26394.970020642686, 26384.43528214425, 26377.33575271535, 26373.697936017845, 26373.548335713524, 26376.91345546425, 26383.819798931872, 26394.29386977818, 26408.362171665045, 26426.051208254306, 26447.387483207778, 26472.397500187282, 26501.10776285467, 26533.544774871814, 26569.735039900494, 26609.70506160255, 26653.48134363984, 26701.090389674173, 26752.558703367416, 26807.912788381385, 26867.1791483779, 26930.384287018845, 26997.554707966003, 27068.71691488124, 27143.89741142637, 27223.122701263215, 27306.41928805365, 27393.81367545948, 27485.332367142524, 27581.001866764687, 27680.848677987735, 27784.89930447354, 27893.18024988391, 28005.71801788069, 28122.539112125727, 28243.67003628081, 28369.137294007865, 28498.967388968646, 28633.186824825003, 28771.82210523879, 28914.899733871818, 29062.446214385956, 29214.48805044299, 29371.051745704797, 29532.163803833144, 29697.85072848999, 29868.139023337113, 30043.05519203627, 30222.625738249342, 30406.87716563823, 30595.835977864743, 30789.528678590606, 30987.98177147776, 31191.221760188084, 31399.275148383305, 31612.16843972526, 31829.928137875882, 32052.580746496904, 32280.152769250257, 32512.670709797632, 32750.16107180105, 32992.650358922154, 33240.16507482297, 33492.73172316515, 33750.37680761066]
test_arr = [61288.20634755, 64510.15236052, 62195.97579225, 69932.40739482, 70852.42048199, 62428.99514288, 69211.23494281, 62947.4127453, 65205.85459849, 67126.11362155, 52328.99273458, 65151.27753168, 61424.54978796, 54996.63373908, 69600.0820475, 59785.08342612, 65919.3096029, 66363.6938435, 65277.59568508, 68189.14144821, 67970.20171678, 65920.53117509, 71914.5527341, 66350.68581769, 66681.5962318, 65851.25664754, 65113.91090946, 58253.75249877, 76720.57027492, 56989.12367932, 69154.47794426, 66825.34474742, 65381.47961272, 63759.53922218, 58975.92369764, 72902.46931747, 72497.97134317, 62894.26574122, 62747.90493878, 62951.99307445, 66873.08998733, 67643.4649575, 70306.94402811, 62707.49384653, 73769.37929999, 67403.15078231, 63018.39532944, 72989.2466467, 61489.27521458, 59027.57151932, 62106.32653734, 63510.81891098, 70380.42081225, 62342.24712848, 67602.99282883, 66581.36814638, 65661.59261251, 62803.46116802, 65821.86331353, 72053.60938622, 71592.77474034, 69552.20701854, 58965.87070429, 61260.93796332, 67359.59680922, 66363.61734578, 71011.26151915, 65969.13652871, 66334.96345182, 72578.35464772, 69877.97617193, 68466.10987673, 65351.38736972, 61088.07458565, 69849.05400386, 63546.7123686, 62603.58775346, 70016.48243305, 59949.92001573, 75763.51629786, 72020.55637112, 65449.37962449, 64086.51359802, 62824.90260637, 62547.02283239, 60360.62866887, 63941.72137617, 72130.24519084, 65880.31444056, 64509.15354963, 52952.67157502, 55253.90955918, 60080.76001801, 65544.91350853, 62262.89263723, 63097.23890012, 62959.14225443, 63526.90828082, 62289.34581955, 67730.48176822, 64012.15564944, 62724.89829366, 57498.29987521, 61206.70470927, 64473.45407499, 70660.8793458, 60997.15642663, 57819.86505662, 49208.61829425, 58605.45612062, 60679.23591514, 60823.1220222, 68306.14000169, 58068.56501041, 52480.54585827, 67626.50443844, 61679.52752567, 59487.99224693, 68419.50499538, 58408.4342905, 65423.47368195, 57026.67025758, 59700.90361682, 55975.52188661, 58724.41692667, 67416.35966403, 63812.31585563, 63690.77757731, 53845.42493582, 54756.11468838, 59499.12132038, 57344.67423713, 59109.73454335, 62029.06506863, 54141.14609801, 58447.45938294, 56994.88607252, 60041.10139828, 66240.16522775, 55011.71476177, 50016.77272235, 65048.92898281, 55838.37765966, 50296.08740919, 49232.47593975, 53895.8071204, 51960.67556055, 49704.8064441, 37446.0043412, 41631.35053158, 52889.96927239, 50623.12446253, 52956.05853793, 55249.15620978, 52109.48657602, 44563.74346679, 50560.01721787, 51969.05577805, 54067.75560275, 51219.67884056, 44169.41772948, 51957.28534285, 38989.98449648, 46619.42235185, 61034.37693728, 59475.46982743, 64623.72564198, 44448.43111426, 47358.88893876, 47743.54378666, 49328.37697084, 50107.35830809, 48931.82221357, 45384.60804015, 34662.97223313, 44930.54806297, 46039.08119679, 39585.27215015, 50047.56016601, 50842.26434552, 34979.50022454, 44131.40308944, 35379.74515717, 39662.2199237, 43239.44452329, 29681.10025659, 40623.67929838, 44170.28131606, 43843.11619687, 41815.23864747, 42698.997335, 43221.95625281, 41501.23823987, 45940.01828022, 28659.63473424, 41976.95839156, 49617.88418777, 47484.74534387, 35073.97538546, 40483.74082902, 42416.72149148, 39348.45836265, 35179.97964377, 45138.75090265, 39454.526969, 44143.95996307, 36251.35765291, 29610.3555882, 41281.95953932, 27411.57568522, 36477.70843025, 41675.85010869, 53060.64437655, 38092.38790211, 48904.7440446, 22725.22099385, 34080.53469906, 36464.453007, 36120.63198101, 43931.77931566, 40795.92515294, 36114.10104668, 37312.39718555, 36082.77619137, 32673.4086326, 35604.82090107, 33387.45029237, 25529.46473164, 30570.04648141, 29311.75291943, 35333.47612653, 33201.09881279, 33662.87026877, 23942.51489714, 22912.25008799, 36503.40532202, 31275.68556357, 44500.1197332, 33269.60260235, 18715.88321171, 44301.98938085, 31096.33666332, 28444.62474096, 32332.55504448, 25017.16802755, 24845.85702745, 36809.41516857, 29435.92796538, 31191.6539642, 32431.3826363, 35968.71404798, 29853.46849567, 27195.0373992, 27732.57272008, 29126.62961724, 35364.51689156, 36176.18692408, 30193.95923488, 31742.96434823, 39534.79473388, 24783.84946831, 23158.17889631, 27957.37555834, 29349.53265991, 18142.54817237, 33319.60393857, 28722.50750299, 28060.41336828, 17670.57271798, 20415.90980139, 21632.42927392, 32870.1930412, 24084.2310184, 28276.9911992, 21406.32662825, 33544.62423838, 26936.59799232, 23214.69544904, 27487.78656074, 28722.37480522, 28608.43519604, 32687.65440296, 22131.70156866, 31929.85276474, 22471.11218259, 28913.21084898, 22419.31411732, 34400.07881496, 26826.30699891, 27902.63561366, 30909.70637316, 32301.82167768, 37257.34343985, 24506.42177572, 25973.24477363, 33011.08766668, 24787.02263347, 27014.85136111, 26255.8112313, 22864.04029881, 30645.72972717, 20095.51722631, 27475.57630632, 30492.40696413, 26181.95133827, 24077.06130541, 24116.32230634, 38946.69898351, 27515.76051837, 38595.98374992, 25984.32769037, 25821.32921507, 28947.43519973, 21614.88706012, 27429.6332694, 31247.40718977, 42336.12141064, 31084.96755089, 27956.59941303, 30123.00824694, 22164.21695465, 26868.41871672, 27554.85610738, 30116.85661126, 25011.79736151, 25102.70904826, 32933.41427988, 26750.82957785, 21505.89372211, 28824.89738001, 28087.04666859, 26188.23251967, 22331.01946436, 26835.44165683, 19293.55930166, 25368.27489596, 32077.17393405, 35379.70259236, 29363.5832013, 26167.04203442, 29878.63239975, 33312.85834696, 30155.0478852, 20358.20864151, 26044.74857795, 33760.79448508, 18502.99408447, 33222.50331648, 26088.92508985, 43204.99182768, 31281.5312429, 37709.62789288, 23776.62386695, 33542.23685627, 30623.25835909, 35411.46256295, 30058.15400615, 49206.62615087, 34578.68265823, 42397.66986141, 32168.72120622, 37730.97545953, 32817.51490677, 26141.65828337, 38101.51374161, 36142.84905364]
cleared_test_arr = mnk_modified(test_arr, 5, 0.0015, 3)
# 8, 0.0012, 3
# 15, 0.0009, 3
# 5 0.0015, 3


plot_two_charts(np.arange(len(test_arr)), test_arr, cleared_test_arr, 'test data', 'cleared')
# plot_two_charts(np.arange(len(test_arr)), cleared_test_arr, test_model, 'cleared', 'model')
# plot_chart(cleared_test_arr, "cleared data")






# Вхід у програму
if __name__ == '__main__':
    main()
