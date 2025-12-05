import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import distance_matrix
from collections import Counter

# Загрузка данных
X, y = load_iris(return_X_y=True)
X = X[:, :2]  # Для наглядности возьмем только первые два признака

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Оптимизированный KNN с векторизацией
def knn_vectorized(x_train, y_train, x_test, k):
    """
    Векторизованная версия KNN без циклов
    
    Параметры:
    -----------
    x_train : numpy array, shape (n_train, n_features)
        Обучающие данные
    y_train : numpy array, shape (n_train,)
        Метки обучающих данных
    x_test : numpy array, shape (n_test, n_features)
        Тестовые данные
    k : int
        Количество соседей
    
    Возвращает:
    -----------
    predictions : numpy array, shape (n_test,)
        Предсказанные метки
    """
    # 1. Вычисляем матрицу расстояний между всеми тестовыми и обучающими точками
    # Используем broadcasting для векторизованного вычисления
    # Формула: ||A - B||^2 = ||A||^2 + ||B||^2 - 2*A·B
    x_train_norm = np.sum(x_train**2, axis=1)  # ||x_train||^2
    x_test_norm = np.sum(x_test**2, axis=1)    # ||x_test||^2
    
    # Расстояния через broadcast: sqrt(||x_test||^2 + ||x_train||^2 - 2*x_test·x_train.T)
    distances = np.sqrt(
        x_test_norm[:, np.newaxis] + 
        x_train_norm[np.newaxis, :] - 
        2 * np.dot(x_test, x_train.T)
    )
    
    # 2. Находим индексы k ближайших соседей для каждой тестовой точки
    # argsort возвращает индексы отсортированных расстояний
    k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
    
    # 3. Получаем метки k ближайших соседей
    k_nearest_labels = y_train[k_nearest_indices]
    
    # 4. Выбираем наиболее частую метку для каждой тестовой точки
    predictions = np.zeros(x_test.shape[0], dtype=y_train.dtype)
    
    for i in range(len(x_test)):
        # Используем Counter для подсчета частоты меток
        counts = Counter(k_nearest_labels[i])
        predictions[i] = counts.most_common(1)[0][0]
    
    return predictions

# Еще более оптимизированная версия с использованием np.bincount
def knn_optimized(x_train, y_train, x_test, k):
    """
    Оптимизированная версия KNN с использованием np.bincount
    
    Параметры:
    -----------
    x_train : numpy array, shape (n_train, n_features)
        Обучающие данные
    y_train : numpy array, shape (n_train,)
        Метки обучающих данных
    x_test : numpy array, shape (n_test, n_features)
        Тестовые данные
    k : int
        Количество соседей
    
    Возвращает:
    -----------
    predictions : numpy array, shape (n_test,)
        Предсказанные метки
    """
    # Преобразуем метки в целые числа от 0 до n_classes-1
    unique_labels = np.unique(y_train)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    y_train_idx = np.array([label_to_idx[label] for label in y_train])
    n_classes = len(unique_labels)
    
    # Вычисляем расстояния с помощью broadcasting
    x_train_norm = np.sum(x_train**2, axis=1)
    x_test_norm = np.sum(x_test**2, axis=1)
    
    distances = np.sqrt(
        x_test_norm[:, np.newaxis] + 
        x_train_norm[np.newaxis, :] - 
        2 * np.dot(x_test, x_train.T)
    )
    
    # Получаем индексы k ближайших соседей
    k_nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]
    
    # Получаем метки ближайших соседей (в виде индексов)
    k_nearest_labels_idx = y_train_idx[k_nearest_indices]
    
    # Подсчитываем голоса для каждого класса
    predictions_idx = np.zeros(x_test.shape[0], dtype=int)
    
    for i in range(len(x_test)):
        # Используем bincount для быстрого подсчета
        votes = np.bincount(k_nearest_labels_idx[i], minlength=n_classes)
        predictions_idx[i] = np.argmax(votes)
    
    # Преобразуем обратно к исходным меткам
    predictions = np.array([idx_to_label[idx] for idx in predictions_idx])
    
    return predictions

# Взвешенная версия KNN с векторизацией
def knn_weighted_vectorized(x_train, y_train, x_test, k, weight_type='inverse'):
    """
    Векторизованная взвешенная версия KNN
    
    Параметры:
    -----------
    weight_type : str
        Тип весовой функции:
        - 'uniform': равные веса
        - 'inverse': обратное расстояние
        - 'inverse_square': обратный квадрат расстояния
        - 'gaussian': гауссовское ядро
    """
    # Вычисляем матрицу расстояний
    x_train_norm = np.sum(x_train**2, axis=1)
    x_test_norm = np.sum(x_test**2, axis=1)
    
    distances_squared = (
        x_test_norm[:, np.newaxis] + 
        x_train_norm[np.newaxis, :] - 
        2 * np.dot(x_test, x_train.T)
    )
    
    # Берем квадратный корень для евклидовых расстояний
    distances = np.sqrt(np.maximum(distances_squared, 0))
    
    # Находим индексы k ближайших соседей
    k_nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]
    
    # Получаем расстояния до k ближайших соседей
    row_indices = np.arange(len(x_test))[:, np.newaxis]
    k_nearest_distances = distances[row_indices, k_nearest_indices]
    
    # Получаем метки k ближайших соседей
    k_nearest_labels = y_train[k_nearest_indices]
    
    # Применяем весовую функцию
    if weight_type == 'uniform':
        weights = np.ones_like(k_nearest_distances)
    elif weight_type == 'inverse':
        weights = 1.0 / (k_nearest_distances + 1e-10)
    elif weight_type == 'inverse_square':
        weights = 1.0 / (k_nearest_distances**2 + 1e-10)
    elif weight_type == 'gaussian':
        # Автоматический выбор sigma (среднее расстояние)
        sigma = np.mean(k_nearest_distances) + 1e-10
        weights = np.exp(-k_nearest_distances**2 / (2 * sigma**2))
    else:
        raise ValueError(f"Неизвестный тип весов: {weight_type}")
    
    # Получаем уникальные метки классов
    unique_labels = np.unique(y_train)
    n_classes = len(unique_labels)
    
    # Создаем словарь для преобразования меток в индексы
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y_train_idx = np.array([label_to_idx[label] for label in y_train])
    
    # Получаем индексы меток для ближайших соседей
    k_nearest_labels_idx = y_train_idx[k_nearest_indices]
    
    # Вычисляем взвешенные голоса для каждого класса
    predictions_idx = np.zeros(len(x_test), dtype=int)
    
    for i in range(len(x_test)):
        # Создаем массив для подсчета взвешенных голосов
        weighted_votes = np.zeros(n_classes)
        
        # Добавляем веса для каждой метки
        for j in range(k):
            label_idx = k_nearest_labels_idx[i, j]
            weighted_votes[label_idx] += weights[i, j]
        
        # Выбираем класс с максимальным весом
        predictions_idx[i] = np.argmax(weighted_votes)
    
    # Преобразуем индексы обратно в метки
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}
    predictions = np.array([idx_to_label[idx] for idx in predictions_idx])
    
    return predictions

# Функция для оценки точности
def accuracy(pred, y):
    return np.mean(pred == y)

# Тестирование и сравнение производительности
def compare_knn_implementations():
    """
    Сравнение производительности разных реализаций KNN
    """
    print("=" * 70)
    print("СРАВНЕНИЕ РЕАЛИЗАЦИЙ KNN")
    print("=" * 70)
    
    # Тестируем на разном количестве данных
    sizes = [50, 100, 200, len(X_train)]
    k = 5
    
    for size in sizes:
        if size > len(X_train):
            continue
            
        # Выбираем подмножество данных
        indices = np.random.choice(len(X_train), size, replace=False)
        x_subset = X_train[indices]
        y_subset = y_train[indices]
        
        print(f"\nРазмер данных: {size}")
        print("-" * 40)
        
        # Оригинальная реализация (с циклами)
        import time
        
        # Векторизованная версия
        start = time.time()
        y_pred_vec = knn_vectorized(x_subset, y_subset, X_test[:10], k)
        vec_time = time.time() - start
        vec_acc = accuracy(y_pred_vec, y_test[:10])
        
        # Оптимизированная версия
        start = time.time()
        y_pred_opt = knn_optimized(x_subset, y_subset, X_test[:10], k)
        opt_time = time.time() - start
        opt_acc = accuracy(y_pred_opt, y_test[:10])
        
        # Взвешенная векторизованная версия
        start = time.time()
        y_pred_weighted = knn_weighted_vectorized(x_subset, y_subset, X_test[:10], k, 'inverse')
        weighted_time = time.time() - start
        weighted_acc = accuracy(y_pred_weighted, y_test[:10])
        
        print(f"Векторизованная KNN:  {vec_time:.6f} сек, точность: {vec_acc:.4f}")
        print(f"Оптимизированная KNN:  {opt_time:.6f} сек, точность: {opt_acc:.4f}")
        print(f"Взвешенная KNN:       {weighted_time:.6f} сек, точность: {weighted_acc:.4f}")
        print(f"Ускорение: {opt_time/vec_time if vec_time > 0 else 0:.2f}x")

# Функция для визуализации границ решений
def visualize_knn_boundaries(x_train, y_train, k, implementation='vectorized'):
    """
    Визуализация границ решений для разных реализаций KNN
    """
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    h = .1
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Выбираем реализацию
    if implementation == 'vectorized':
        Z = knn_vectorized(x_train, y_train, grid_points, k)
    elif implementation == 'optimized':
        Z = knn_optimized(x_train, y_train, grid_points, k)
    elif implementation == 'weighted':
        Z = knn_weighted_vectorized(x_train, y_train, grid_points, k, 'inverse')
    
    Z = np.array(Z).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap, 
                edgecolor='black', s=50)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"{implementation.capitalize()} KNN (k={k})", fontsize=14)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.show()

# Основная программа
if __name__ == "__main__":
    print("Демонстрация оптимизированных реализаций KNN")
    print("=" * 60)
    
    k = 5
    
    # Тестируем разные реализации
    print("\n1. Тестирование на тестовой выборке:")
    print("-" * 40)
    
    # Векторизованная версия
    y_pred_vec = knn_vectorized(X_train, y_train, X_test, k)
    acc_vec = accuracy(y_pred_vec, y_test)
    print(f"Векторизованная KNN: точность = {acc_vec:.4f}")
    
    # Оптимизированная версия
    y_pred_opt = knn_optimized(X_train, y_train, X_test, k)
    acc_opt = accuracy(y_pred_opt, y_test)
    print(f"Оптимизированная KNN: точность = {acc_opt:.4f}")
    
    # Взвешенная версия
    y_pred_weighted = knn_weighted_vectorized(X_train, y_train, X_test, k, 'inverse')
    acc_weighted = accuracy(y_pred_weighted, y_test)
    print(f"Взвешенная KNN: точность = {acc_weighted:.4f}")
    
    # Сравнение производительности
    compare_knn_implementations()
    
    # Визуализация границ решений
    print("\n2. Визуализация границ решений:")
    print("-" * 40)
    
    visualize_knn_boundaries(X_train, y_train, k, 'vectorized')
    visualize_knn_boundaries(X_train, y_train, k, 'optimized')
    visualize_knn_boundaries(X_train, y_train, k, 'weighted')
    
    # Дополнительный бенчмарк
    print("\n3. Бенчмарк на больших данных:")
    print("-" * 40)
    
    # Создаем искусственные большие данные
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    X_large = np.random.randn(n_samples, n_features)
    y_large = np.random.randint(0, 3, n_samples)
    
    # Разделяем на обучающую и тестовую выборки
    X_train_large, X_test_large = X_large[:800], X_large[800:]
    y_train_large, y_test_large = y_large[:800], y_large[800:]
    
    print(f"Размер данных: {n_samples} объектов, {n_features} признаков")
    
    import time
    
    # Тестируем оптимизированную версию
    start = time.time()
    y_pred_large = knn_optimized(X_train_large, y_train_large, X_test_large[:50], k)
    time_taken = time.time() - start
    
    print(f"Время предсказания для 50 объектов: {time_taken:.4f} сек")
    print(f"Среднее время на объект: {time_taken/50:.6f} сек")