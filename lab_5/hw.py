import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def tree_vote(forest, data):
    predictions = []
    for tree in forest:
        predictions.append(tree.predict(data))
    
    # сформируем список с предсказаниями для каждого объекта
    predictions_per_object = list(zip(*predictions))
    
    # выберем в качестве итогового предсказания для каждого объекта то,
    # за которое проголосовало большинство деревьев
    voted_predictions = []
    for obj in predictions_per_object:
        voted_predictions.append(max(set(obj), key=obj.count))
        
    return voted_predictions

def plot_decision_boundary(forest, X, y, title, ax):
    """
    Функция для визуализации разделяющей гиперплоскости с использованием tree_vote
    """
    # Создаем сетку для построения графика
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = tree_vote(forest, grid_points)
    Z = np.array(Z).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                        edgecolors='k', s=20)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')

# Создаем датасет
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

print(f"Размерность данных: {X.shape}")
print(f"Количество классов: {len(np.unique(y))}")

# Список количества деревьев для экспериментов
n_trees_list = [1, 3, 10, 50]

# Создаем subplot для визуализации
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, n_trees in enumerate(n_trees_list):
    print(f"Обучаем случайный лес с {n_trees} деревьями...")
    
    # Создаем и обучаем случайный лес
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(X, y)
    
    # Визуализируем разделяющую гиперплоскость с использованием tree_vote
    plot_decision_boundary(rf.estimators_, X, y, 
                          f'Случайный лес ({n_trees} дерево(ьев))', 
                          axes[i])
    
    # Вычисляем точность с помощью tree_vote
    voted_predictions = tree_vote(rf.estimators_, X)
    accuracy = np.mean(voted_predictions == y)
    
    # Добавляем информацию о точности
    axes[i].text(0.05, 0.95, f'Точность: {accuracy:.3f}', 
                transform=axes[i].transAxes, fontsize=12,
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

plt.tight_layout()
plt.suptitle('Визуализация разделяющих гиперплоскостей случайного леса\n(с использованием tree_vote)', 
             fontsize=14, y=1.02)
plt.show()
