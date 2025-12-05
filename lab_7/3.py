import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Загрузка данных
X, y = load_iris(return_X_y=True)
X = X[:, :2]  # Берем только первые два признака для наглядности

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Функция евклидова расстояния
def e_metrics(x1, x2):
    distance = np.sum(np.square(x1 - x2))
    return np.sqrt(distance)

# Алгоритм k-means
def kmeans(data, k, max_iterations, min_distance):
    clusters = {i: [] for i in range(k)}
    centroids = [data[i] for i in range(k)]
    
    for _ in range(max_iterations):
        # Очищаем кластеры перед новым распределением
        clusters = {i: [] for i in range(k)}
        
        # Распределяем точки по кластерам
        for x in data:
            distances = [e_metrics(x, centroid) for centroid in centroids]
            cluster = distances.index(min(distances))
            clusters[cluster].append(x)
        
        # Сохраняем старые центроиды
        old_centroids = centroids.copy()
        
        # Пересчитываем центроиды
        for cluster in clusters:
            if clusters[cluster]:  # Проверяем, что кластер не пустой
                centroids[cluster] = np.mean(clusters[cluster], axis=0)
            else:
                # Если кластер пустой, оставляем старый центроид
                centroids[cluster] = old_centroids[cluster]
        
        # Проверяем сходимость
        optimal = True
        for centroid in range(len(centroids)):
            if np.linalg.norm(centroids[centroid] - old_centroids[centroid], ord=2) > min_distance:
                optimal = False
                break
        
        if optimal:
            break
    
    return centroids, clusters

# Функция для подсчета среднего квадратичного внутрикластерного расстояния
def calculate_within_cluster_sse(data, clusters, centroids):
    """
    Вычисляет среднее квадратичное внутрикластерное расстояние (within-cluster SSE)
    
    Parameters:
    -----------
    data : numpy array
        Исходные данные
    clusters : dict
        Словарь кластеров {cluster_id: [точки]}
    centroids : list
        Список центроидов
    
    Returns:
    --------
    float : среднее квадратичное внутрикластерное расстояние
    """
    total_sse = 0.0
    total_points = 0
    
    for cluster_id in clusters:
        if clusters[cluster_id]:
            points = np.array(clusters[cluster_id])
            centroid = centroids[cluster_id]
            
            # Вычисляем сумму квадратов расстояний от точек до центроида кластера
            distances_squared = np.sum((points - centroid) ** 2)
            total_sse += distances_squared
            total_points += len(points)
    
    # Среднее квадратичное расстояние (нормируем на количество точек)
    if total_points > 0:
        return total_sse / total_points
    else:
        return 0.0

# Анализ зависимости метрики от количества кластеров k
def analyze_k_range(data, k_range, max_iterations=100, min_distance=0.001):
    """
    Анализирует зависимость внутрикластерного расстояния от количества кластеров
    
    Parameters:
    -----------
    data : numpy array
        Исходные данные
    k_range : range or list
        Диапазон значений k для анализа
    max_iterations : int
        Максимальное количество итераций для k-means
    min_distance : float
        Минимальное расстояние для остановки алгоритма
    
    Returns:
    --------
    dict : словарь с результатами {k: within_cluster_sse}
    """
    results = {}
    
    for k in k_range:
        # Выполняем k-means кластеризацию
        centroids, clusters = kmeans(data, k, max_iterations, min_distance)
        
        # Вычисляем метрику качества
        within_sse = calculate_within_cluster_sse(data, clusters, centroids)
        results[k] = within_sse
        
        print(f"k = {k}: среднее квадратичное внутрикластерное расстояние = {within_sse:.4f}")
    
    return results

# Функция для построения графика "локтевого метода" (elbow method)
def plot_elbow_method(results):
    """
    Строит график зависимости внутрикластерного расстояния от количества кластеров
    
    Parameters:
    -----------
    results : dict
        Словарь с результатами {k: within_cluster_sse}
    """
    k_values = list(results.keys())
    sse_values = list(results.values())
    
    plt.figure(figsize=(12, 6))
    
    # Основной график
    plt.subplot(1, 2, 1)
    plt.plot(k_values, sse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Количество кластеров (k)', fontsize=12)
    plt.ylabel('Среднее квадратичное внутрикластерное расстояние', fontsize=12)
    plt.title('Зависимость метрики качества от количества кластеров', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Добавляем точки на график
    for k, sse in zip(k_values, sse_values):
        plt.text(k, sse + 0.02, f'{sse:.3f}', ha='center', va='bottom', fontsize=10)
    
    # График производной (скорость уменьшения ошибки)
    plt.subplot(1, 2, 2)
    if len(k_values) > 1:
        # Вычисляем разности между соседними значениями SSE
        diff_values = [0]  # Для k=1 разность равна 0
        for i in range(1, len(sse_values)):
            diff = sse_values[i-1] - sse_values[i]
            diff_values.append(diff)
        
        plt.bar(k_values, diff_values, alpha=0.7, color='orange')
        plt.xlabel('Количество кластеров (k)', fontsize=12)
        plt.ylabel('Уменьшение SSE при увеличении k', fontsize=12)
        plt.title('Скорость уменьшения внутрикластерной дисперсии', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for k, diff in zip(k_values, diff_values):
            if diff > 0:
                plt.text(k, diff + 0.002, f'{diff:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return k_values, sse_values

# Функция для визуализации кластеризации для разных k
def visualize_clusters_for_k(data, k_values, max_iterations=100, min_distance=0.001):
    """
    Визуализирует результаты кластеризации для разных значений k
    
    Parameters:
    -----------
    data : numpy array
        Исходные данные
    k_values : list
        Список значений k для визуализации
    max_iterations : int
        Максимальное количество итераций
    min_distance : float
        Минимальное расстояние для остановки
    """
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(5*n_k, 5))
    
    if n_k == 1:
        axes = [axes]
    
    colors = ['r', 'g', 'b', 'orange', 'y', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    for idx, k in enumerate(k_values):
        # Выполняем кластеризацию
        centroids, clusters = kmeans(data, k, max_iterations, min_distance)
        
        # Вычисляем метрику качества
        within_sse = calculate_within_cluster_sse(data, clusters, centroids)
        
        # Визуализируем
        ax = axes[idx]
        
        # Рисуем точки кластеров
        for cluster_id in clusters:
            if clusters[cluster_id]:
                points = np.array(clusters[cluster_id])
                color_idx = cluster_id % len(colors)
                ax.scatter(points[:, 0], points[:, 1], 
                          color=colors[color_idx], 
                          alpha=0.6, 
                          label=f'Кластер {cluster_id}')
        
        # Рисуем центроиды
        for centroid in centroids:
            ax.scatter(centroid[0], centroid[1], 
                      marker='X', s=150, c='black', linewidths=2)
        
        ax.set_title(f'k = {k}\nSSE = {within_sse:.3f}', fontsize=12)
        ax.set_xlabel('Признак 1')
        if idx == 0:
            ax.set_ylabel('Признак 2')
        ax.grid(True, alpha=0.3)
        # ax.legend()
    
    plt.suptitle('Визуализация кластеризации для разных k', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    # Используем обучающую выборку для анализа
    data = X_train

    
    # Анализируем для k от 1 до 10
    k_range = range(1, 11)
    
    results = analyze_k_range(data, k_range, max_iterations=100, min_distance=0.001)

    
    # Находим оптимальное k по "локтевому методу"
    k_values = list(results.keys())
    sse_values = list(results.values())
    
    # Вычисляем "выгоду" от увеличения k
    improvements = []
    for i in range(1, len(sse_values)):
        improvement = (sse_values[i-1] - sse_values[i]) / sse_values[i-1] * 100
        improvements.append(improvement)
    
    if improvements:
        # Находим точку, где улучшение становится меньше 10%
        optimal_k = 2
        for i, improvement in enumerate(improvements):
            if improvement < 10 and i > 0:
                optimal_k = k_values[i]
                break
        
        print(f"\nРекомендуемое количество кластеров (по локтевому методу): k = {optimal_k}")
    
    
    # Строим графики
    plot_elbow_method(results)
    
    # Визуализируем кластеризацию для некоторых k
    selected_k = [2, 3, 4, 5]
    visualize_clusters_for_k(data, selected_k)
    

    # Выполняем кластеризацию для k=3
    centroids, clusters = kmeans(data, k=3, max_iterations=100, min_distance=0.001)
    
    # Сравниваем с истинными метками
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Создаем предсказанные метки кластеров
    y_pred = np.zeros(len(data))
    for cluster_id in clusters:
        for point in clusters[cluster_id]:
            # Находим индекс точки в исходных данных
            point_idx = np.where((data == point).all(axis=1))[0]
            if len(point_idx) > 0:
                y_pred[point_idx[0]] = cluster_id
    
    # Используем соответствующие истинные метки для обучающей выборки
    true_labels = y_train
    
    # Вычисляем метрики сравнения кластеризации
    ari = adjusted_rand_score(true_labels, y_pred)
    nmi = normalized_mutual_info_score(true_labels, y_pred)
