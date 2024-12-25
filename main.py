"""
Виконав: Литвиненко Роман
Лабораторна робота №9, I рівень складності з додатковим завданням(+ візуалізаця для рівня II в кінці)
"""
import matplotlib
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point
matplotlib.use('TkAgg')


def main():
    # Завантаження даних
    fire_stations = gpd.read_file("Fire_Stations/Fire_Stations.shp")

    # Перевірка завантажених даних
    print(fire_stations.head())
    print("Система координат:", fire_stations.crs)

    # Переведення в систему координат EPSG:4326 (WGS 84)
    fire_stations = fire_stations.to_crs("EPSG:4326")

    # Візуалізація початкових даних
    fire_stations.plot(figsize=(10, 8), color="blue", markersize=5)
    plt.title("Розташування пожежних станцій США")
    plt.show()

    # Щільність розташування станцій (на 10 000 км²)
    us_area_km2 = 9834000  # Площа США в км²
    station_density = (len(fire_stations) / us_area_km2) * 10000
    print(f"Щільність пожежних станцій: {station_density:.2f} станцій на 10 000 км²")

    # Вибірка перших 1000 станцій
    fire_stations = fire_stations[:1000].copy()

    # Розрахунок відстаней між усіма станціями
    coordinates = fire_stations.geometry.apply(lambda geom: (geom.x, geom.y))
    coordinates = np.array(list(coordinates))

    # Обчислення матриці відстаней
    distance_matrix = squareform(pdist(coordinates, metric=lambda u, v: haversine(u[1], u[0], v[1], v[0])))

    # Середня відстань між станціями
    n_stations = len(coordinates)
    average_distance = np.sum(distance_matrix) / (n_stations * (n_stations - 1))
    print(f"Середня відстань між пожежними станціями: {average_distance:.2f} км")

    # Визначення центроїдів найбільш густо населених станціями районів
    # Переведення в проєкцію для аналізу на площині
    fire_stations = fire_stations.to_crs("EPSG:5070")

    # Створення регулярної сітки
    grid_size = 50000  # Розмір сітки в метрах
    xmin, ymin, xmax, ymax = fire_stations.total_bounds
    x_bins = np.arange(xmin, xmax, grid_size)
    y_bins = np.arange(ymin, ymax, grid_size)

    fire_stations['x_bin'] = np.digitize(fire_stations.geometry.x, x_bins)
    fire_stations['y_bin'] = np.digitize(fire_stations.geometry.y, y_bins)

    # Групування за сіткою і підрахунок кількості станцій
    densest_regions = fire_stations.groupby(['x_bin', 'y_bin']).size().reset_index(name='count')
    densest_regions = densest_regions.sort_values(by='count', ascending=False)

    # Візуалізація найбільш густо населених районів
    fig, ax = plt.subplots(figsize=(12, 10))
    fire_stations.plot(ax=ax, color="blue", markersize=5, alpha=0.5, label="Станції")

    # Відображення 5 найгустіших районів
    top_regions = densest_regions.head(5)
    for _, row in top_regions.iterrows():
        x_center = x_bins[row['x_bin'] - 1] + grid_size / 2
        y_center = y_bins[row['y_bin'] - 1] + grid_size / 2
        ax.scatter(x_center, y_center, s=100, label=f"Густота: {row['count']}")

    plt.title("Густонаселені райони пожежних станцій")
    plt.legend()
    plt.show()

    # Перевірка системи координат і одиниць виміру
    print("Перевірка системи координат:", fire_stations.crs)
    print("Одиниці виміру: метри (використовується EPSG:5070 для аналізу)")

    print('Група завдань 2 рівня')
    level2_try()


def haversine(lat1, lon1, lat2, lon2):
    """
    Формула Гаверсина для обчислення великоколової відстані між двома точками
    """
    R = 6371  # Радіус Землі в км
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def level2_try():
    """
    Тут я намагався виконати завдання, проте, скоріш за все, через проблеми зі вхідними даними, нічого не вийшло
    окрім візуалізації карти
    """
    # Завантаження даних
    csv_file = 'ukr_pd_2020_1km_ASCII_XYZ.csv'  # Replace with your CSV file path
    data = pd.read_csv(csv_file)
    print(1)

    # Перетворення даних на GeoDataFrame
    data['geometry'] = data.apply(lambda row: Point(row['X'], row['Y']), axis=1)
    geo_df = gpd.GeoDataFrame(data, geometry='geometry')
    print(2)

    # Встановдення системи координат
    geo_df.set_crs(epsg=4326, inplace=True)
    print(3)

    # Ввід карти
    geo_df.plot(figsize=(10, 10), color='blue', markersize=50, legend=True)
    plt.title("Мапа щільності населення")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
