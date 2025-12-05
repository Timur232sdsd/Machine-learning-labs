import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Загрузка данных
X, y = load_iris(return_X_y=True)
X = X[:, :2]  # Для наглядности возьмем только первые два признака
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Евклидова метрика расстояния
def e_metrics(x1, x2):
    distance = np.sum(np.square(x1 - x2))
    return np.sqrt(distance)

# Модифицированный алгоритм kNN с весами
def knn_weighted(x_train, y_train, x_test, k, weight_type='uniform'):
    """
    Реализация kNN с весами
    
    Параметры:
    -----------
    weight_type : str
        Тип весовой функции:
        - 'uniform': равные веса (стандартный kNN)
        - 'inverse': обратное расстояние (1 / distance)
        - 'inverse_square': обратный квадрат расстояния (1 / distance^2)
        - 'gaussian': гауссовское ядро exp(-distance^2 / sigma^2)
    """
    answers = []
    
    for x in x_test:
        test_distances = []
        
        # Рассчитываем расстояния до всех объектов обучающей выборки
        for i in range(len(x_train)):
            distance = e_metrics(x, x_train[i])
            test_distances.append((distance, y_train[i]))
        
        # Сортируем по расстоянию и берем k ближайших соседей
        neighbors = sorted(test_distances)[0:k]
        
        # Создаем словарь для подсчета взвешенных голосов
        classes = {class_item: 0.0 for class_item in set(y_train)}
        
        # Применяем весовую функцию
        for distance, class_label in neighbors:
            if weight_type == 'uniform':
                weight = 1.0
            elif weight_type == 'inverse':
                # Добавляем маленькое значение для избежания деления на 0
                weight = 1.0 / (distance + 1e-10)
            elif weight_type == 'inverse_square':
                weight = 1.0 / (distance**2 + 1e-10)
            elif weight_type == 'gaussian':
                # Автоматический подбор sigma (можно настроить)
                sigma = np.mean([d for d, _ in neighbors]) + 1e-10
                weight = np.exp(-distance**2 / (2 * sigma**2))
            else:
                raise ValueError(f"Неизвестный тип весов: {weight_type}")
            
            classes[class_label] += weight
        
        # Выбираем класс с наибольшим суммарным весом
        answers.append(max(classes, key=classes.get))
    
    return answers

# Функция для оценки точности
def accuracy(pred, y):
    return sum(pred == y) / len(y)

# Тестирование с разными типами весов
k = 5
weight_types = ['uniform', 'inverse', 'inverse_square', 'gaussian']


for weight_type in weight_types:
    y_pred = knn_weighted(X_train, y_train, X_test, k, weight_type)
    acc = accuracy(y_pred, y_test)
    print(f"Тип весов: {weight_type:15} Точность: {acc:.4f}")

# Визуализация с весами
def get_graph_weighted(X_train, y_train, k, weight_type='inverse'):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    h = .1
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Используем взвешенный kNN для предсказаний
    Z = knn_weighted(X_train, y_train, np.c_[xx.ravel(), yy.ravel()], k, weight_type)
    
    # Построение графика
    Z = np.array(Z).reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    # Добавление обучающей выборки
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor='black', s=50)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Взвешенный kNN (k={k}, веса={weight_type})")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.show()

# Пример визуализации с обратными весами
get_graph_weighted(X_train, y_train, k=5, weight_type='inverse')

k_values = [1, 3, 5, 7, 9]
weight_types = ['uniform', 'inverse', 'inverse_square']

# Таблица результатов
results = np.zeros((len(k_values), len(weight_types)))

for i, k in enumerate(k_values):
    for j, weight_type in enumerate(weight_types):
        y_pred = knn_weighted(X_train, y_train, X_test, k, weight_type)
        results[i, j] = accuracy(y_pred, y_test)

# Визуализация результатов
plt.figure(figsize=(10, 6))
for j, weight_type in enumerate(weight_types):
    plt.plot(k_values, results[:, j], marker='o', label=weight_type, linewidth=2)

plt.xlabel('Количество соседей (k)')
plt.ylabel('Точность')
plt.title('Влияние весовой функции на точность kNN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()