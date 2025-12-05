import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X, y = load_iris(return_X_y=True)

# Для наглядности возьмем только первые два признака (всего в датасете их 4)
X = X[:, :2]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)
X_train.shape, X_test.shape

def e_metrics(x1, x2):
    
    distance = np.sum(np.square(x1 - x2))

    return np.sqrt(distance)

{class_item: 0 for class_item in set(y_train)}

def knn(x_train, y_train, x_test, k):
    
    answers = []
    for x in x_test:
        test_distances = []
            
        for i in range(len(x_train)):
            
            # расчет расстояния от классифицируемого объекта до
            # объекта обучающей выборки
            distance = e_metrics(x, x_train[i])
            
            # Записываем в список значение расстояния и ответа на объекте обучающей выборки
            test_distances.append((distance, y_train[i]))
        
        # создаем словарь со всеми возможными классами
        classes = {class_item: 0 for class_item in set(y_train)}
        
        # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
        for d in sorted(test_distances)[0:k]:
            classes[d[1]] += 1

        # Записываем в список ответов наиболее часто встречающийся класс
        answers.append(sorted(classes, key=classes.get)[-1])
        
    return answers

def accuracy(pred, y):
    return (sum(pred == y) / len(y))
k = 2

y_pred = knn(X_train, y_train, X_test, k)

def get_graph(X_train, y_train, k):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])

    h = .1

    # Расчет пределов графика
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(np.c_[xx.ravel(), yy.ravel()].shape)

    # Получим предсказания для всех точек
    Z = knn(X_train, y_train, np.c_[xx.ravel(), yy.ravel()], k)
    # Построим график
    Z = np.array(Z).reshape(xx.shape)
    plt.figure(figsize=(7,7))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Добавим на график обучающую выборку
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Трехклассовая kNN классификация при k = {k}")
    plt.show()

k = 3

y_pred = knn(X_train, y_train, X_test, k)

k = 5

y_pred = knn(X_train, y_train, X_test, k)

x = np.zeros((2000))
y = np.zeros((2000))
y[2] = 100

x = np.zeros((2000))
z = np.ones((2000))

true = np.array([0, 0, 0, 1, 1, 1])
pred = np.array([0, 1, 2, 3, 4, 5])

from sklearn.metrics.cluster import contingency_matrix
cm = contingency_matrix(true, pred)

h_c_k = 0
for j in range(cm.shape[1]):
    for i in range(cm.shape[0]):
        size = np.sum(cm)
        p_c_k = cm[i][j] / size
        p_k = np.sum(cm[:, j]) / size
        if p_c_k == 0:
            continue
        h_c_k += p_c_k * np.log2(p_c_k / p_k)
        
h_c = 0
for i in range(cm.shape[0]):
    size = np.sum(cm)
    p_c = np.sum(cm[i]) / size

    h_c += p_c * np.log2(p_c)
        
h = 1 - h_c_k / h_c

true = np.array([0, 0, 0, 1, 1, 1])
pred = np.array([0, 1, 2, 3, 4, 5])

cm = contingency_matrix(true, pred)

h_k_c = 0
for j in range(cm.shape[0]):
    for i in range(cm.shape[1]):
        size = np.sum(cm)
        p_k_c = cm[j][i] / size
        p_c = np.sum(cm[j]) / size
        
        if p_k_c == 0:
            continue
        h_k_c += p_k_c * np.log2((p_k_c / p_c) + 1e-18)
        
h_k = 0
for i in range(cm.shape[1]):
    size = np.sum(cm)
    p_k = np.sum(cm[:, i]) / size

    h_k += p_k * np.log2(p_k + 1e-18)
    
    
h_k += 1e-18
        
c = 1 - h_k_c/ h_k

import matplotlib.pyplot as plt

def e_metrics(x1, x2):
    
    distance = np.sum(np.square(x1 - x2))

    return np.sqrt(distance)

def kmeans(data, k, max_iterations, min_distance):
    # Создадим словарь для кластеризации
    clusters = {i: [] for i in range(k)}
    
    # инициализируем центроиды как первые k элементов датасета
    centroids = [data[i] for i in range(k)]
    
    for _ in range(max_iterations):
        # кластеризуем объекты по центроидам
        for x in data:
            # определим расстояния от объекта до каждого центроида
            distances = [e_metrics(x, centroid) for centroid in centroids]
            # отнесем объект к кластеру, до центроида которого наименьшее расстояние
            cluster = distances.index(min(distances))
            clusters[cluster].append(x)
        
        # сохраним предыдущие центроиды в отдельный список для последующего сравнения сновыми
        old_centroids = centroids.copy()
        
        # пересчитаем центроиды как среднее по кластерам
        for cluster in clusters:
            centroids[cluster] = np.mean(clusters[cluster], axis=0)
            
        # сравним величину смещения центроидов с минимальной
        optimal = True
        for centroid in range(len(centroids)):
            if np.linalg.norm(centroids[centroid] - old_centroids[centroid], ord=2) > min_distance:
                optimal = False
                break
        
        # если все смещения меньше минимального, останавливаем алгоритм  
        if optimal:
            break
    
    return old_centroids, clusters


def visualize(centroids, clusters):
    colors = ['r', 'g', 'b', 'orange', 'y']
    
    plt.figure(figsize=(7,7))
    
    # нанесем на график центроиды
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], marker='x', s=130, c='black')
        
    # нанесем объекты раскрашенные по классам
    for cluster_item in clusters:
        for x in clusters[cluster_item]:
            plt.scatter(x[0], x[1], color=colors[cluster_item])
            
    plt.show()

