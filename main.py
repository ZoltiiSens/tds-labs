"""
Виконав: Литвиненко Роман
Лабораторна робота №4, III рівень складності (але 4 групи технічних вимог)

"""
import random
import matplotlib
import cv2
import time
import imutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn import svm
from scipy.cluster.hierarchy import dendrogram, linkage
from imutils.video import FPS

matplotlib.use('TkAgg')


def main():
    first_group()
    second_group()
    third_group()
    fifth_group()


def first_group():
    ''' Реалізація завдань групи технічних вимог 1 '''
    def generate_random_values(size, min_value=0, max_value=100):
        ''' Функція геренації випадкових вхідних даних '''
        result_x = []
        result_y = []
        for i in range(size):
            result_x.append(random.random() * max_value + min_value)
            result_y.append(random.random() * max_value + min_value)
        return result_x, result_y

    def read_data_from_excel():
        ''' Функція читання даних з файлу '''
        data = pd.read_excel('group1_data.xlsx')
        print(data)
        return list(data['Current_point']), list(data['Previous_point'])

    print('Група 1 починається...')
    # Вибір даних для роботи
    user_answer = input('Оберіть джерело даних:\n * 0 - випадкові\n * 1 - реальні\nEnter: ')
    if user_answer == '0':
        input_x, input_y = generate_random_values(100)
        x_title = 'values_x'
        y_title = 'values_y'
        title = 'Random data clustering'
    elif user_answer == '1':
        input_x, input_y = read_data_from_excel()
        x_title = 'Current points'
        y_title = 'Old points'
        title = 'Students clustering based on their points for 2 semesters'
    else:
        print('Wrong input!')
        return
    plt.scatter(input_x, input_y)
    plt.title = title
    plt.xlabel = x_title
    plt.ylabel = y_title
    plt.show()

    # Вибір алгоритму кластеризації
    user_answer = input('Оберіть алгоритм кластеризації:\n * 0 - k-means\n * 1 - svm\n * 2 - k-nearest neighbors\n * 3 - ієрархічна кластеризація\nEnter: ')
    if user_answer == '0':
        number_of_clusters = int(input('Введіть кількість кластерів: '))
        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit(list(zip(input_x, input_y)))
        plt.scatter(input_x, input_y, c=kmeans.labels_)
        plt.title = title + ' clustered'
        plt.xlabel = x_title
        plt.ylabel = y_title
        plt.show()
    elif user_answer == '1':
        number_of_clusters = int(input('Введіть кількість кластерів: '))
        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit(list(zip(input_x, input_y)))
        clf = svm.SVC(kernel='linear')
        clf.fit(np.column_stack((input_x, input_y)), kmeans.labels_)
        h = .1
        x_min, x_max = min(input_x) - 1, max(input_x) + 1
        y_min, y_max = min(input_y) - 1, max(input_y) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(input_x, input_y, edgecolors='k', cmap='coolwarm')
        plt.title = title + ' clustered with SVM'
        plt.xlabel = x_title
        plt.ylabel = y_title
        plt.show()
    elif user_answer == '2':
        number_of_clusters = int(input('Введіть кількість кластерів: '))
        number_of_neighbors = int(input('Введіть кількість сусідів: '))
        knn = KNeighborsClassifier(n_neighbors=number_of_neighbors)
        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit(list(zip(input_x, input_y)))
        knn.fit(list(zip(input_x, input_y)), kmeans.labels_)
        new_random_point = [(random.random() * max(input_x) + min(input_x), random.random() * max(input_y) + min(input_y))]
        prediction = knn.predict(new_random_point)
        print(new_random_point)
        plt.scatter(input_x, input_y, c=kmeans.labels_)
        plt.scatter(new_random_point[0][0], new_random_point[0][1], c=prediction[0])
        plt.text(x=new_random_point[0][0] + 3, y=new_random_point[0][1], s=f'new point, predicted class: {prediction[0]}')
        plt.title = title + ' clustered with KNeighbors'
        plt.xlabel = x_title
        plt.ylabel = y_title
        plt.show()
    elif user_answer == '3':
        number_of_clusters = int(input('Введіть кількість кластерів: '))
        data = np.array(list(zip(input_x, input_y)))
        linkage_data = linkage(data, method='ward', metric='euclidean')
        dendrogram(linkage_data)
        plt.show()
        hierarchical_cluster = AgglomerativeClustering(n_clusters=number_of_clusters, linkage='ward')
        labels = hierarchical_cluster.fit_predict(data)
        plt.scatter(input_x, input_y, c=labels)
        plt.show()
    else:
        print('Wrong input!')
        return


def second_group(img_path='test_image.jpg'):
    ''' Реалізація завдань групи технічних вимог 2 '''
    print('Група 2 починається...')
    # Читання зображення
    img = cv2.imread(img_path)
    old_image = img.copy()

    # Кольорова обробка зображення
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Кластеризація
    K = 2
    attempts = 10
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    edged_img = res.reshape(img.shape)

    # Вивід зображень
    plt.subplot(121), plt.imshow(old_image)
    plt.title = 'Input Image'
    plt.subplot(122), plt.imshow(edged_img, 'gray')
    cv2.imwrite('resulted2_' + img_path, edged_img)
    plt.title = "kmeans"
    plt.tight_layout()
    plt.show()
    return


def third_group(img_path="test_image.jpg"):
    ''' Реалізація завдань групи технічних вимог 3 '''
    print('Група 3 починається...')
    # Читаємо зображення
    image = cv2.imread(img_path)
    plt.imshow(image)
    plt.show()

    # Проводимо обробку зображення шляхом заміни кольорів на 2, підвищення контрасту, блюру, фільтрації
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.equalizeHist(processed_img)
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0.45)
    processed_img = cv2.Canny(processed_img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)
    plt.imshow(processed_img, cmap='gray')
    plt.show()

    # Отримання контурів
    contour, _ = cv2.findContours(processed_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Пошук елементів за умовами
    total = 0
    for c in contour:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = max(w, h) / min(w, h)
        # Відкидаємо занадто "довгі" елементи
        if aspect_ratio > 2:
            continue
        # Розглядаємо тільки об'єкти середнього розміру з кількістю кутів 3-9
        if 1200 > area > 400 and len(approx) in (3, 4, 5, 6, 7, 8, 9):
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
            total += 1
    print(f"Знайдено {total} сегментів за умовами")
    cv2.imwrite('resulted3_' + img_path, image)
    plt.imshow(image)
    plt.show()


def fifth_group(video_path='test_video2.mp4'):
    ''' Реалізація завдань групи технічних вимог 5 '''
    print('Група 5 починається...')
    # Завантаження моделі
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", 'MobileNetSSD_deploy.caffemodel')
    cap = cv2.VideoCapture(video_path)
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    time.sleep(0.1)
    fps = FPS().start()
    # Початок обробки відео
    while True:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=900)
        try:
            (h, w) = frame.shape[:2]
        except:
            break
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)),0.007843, (400, 400), 127.5)
        net.setInput(blob)
        # Детекція елементів на фреймі
        detections = net.forward()
        time.sleep(0.009)
        # Вивід результатів детекції
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        fps.update()


if __name__ == '__main__':
    main()
