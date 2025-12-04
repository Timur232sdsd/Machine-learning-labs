from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25, random_state=42
)

def mean_squared_error(y_real, prediction):
    """Среднеквадратичная ошибка"""
    return np.mean((y_real - prediction) ** 2)

# =============================================
# 1. Оригинальный градиентный бустинг
# =============================================
def gb_fit_original(n_trees, max_depth, X_train, X_test, y_train, y_test, eta):
    """Оригинальный градиентный бустинг (полная выборка)"""
    trees = []
    train_errors = []
    test_errors = []
    
    current_prediction_train = np.zeros(len(y_train))
    current_prediction_test = np.zeros(len(y_test))
    
    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        if i == 0:
            tree.fit(X_train, y_train)
        else:
            residuals = y_train - current_prediction_train
            tree.fit(X_train, residuals)
        
        trees.append(tree)
        current_prediction_train += eta * tree.predict(X_train)
        current_prediction_test += eta * tree.predict(X_test)
        
        train_errors.append(mean_squared_error(y_train, current_prediction_train))
        test_errors.append(mean_squared_error(y_test, current_prediction_test))
    
    return train_errors, test_errors

# =============================================
# 2. Стохастический градиентный бустинг
# =============================================
def gb_fit_stochastic(n_trees, max_depth, X_train, X_test, y_train, y_test, eta, subsample=0.5):
    """Стохастический градиентный бустинг (подвыборка)"""
    train_errors = []
    test_errors = []
    
    current_prediction_train = np.zeros(len(y_train))
    current_prediction_test = np.zeros(len(y_test))
    
    n_samples = int(len(y_train) * subsample)
    
    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        # Случайная подвыборка данных
        indices = np.random.choice(len(y_train), size=n_samples, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        
        if i == 0:
            tree.fit(X_batch, y_batch)
        else:
            # Вычисляем остатки на всей выборке, но обучаем на подвыборке
            residuals_full = y_train - current_prediction_train
            residuals_batch = residuals_full[indices]
            tree.fit(X_batch, residuals_batch)
        
        # Обновляем предсказания на всей выборке
        current_prediction_train += eta * tree.predict(X_train)
        current_prediction_test += eta * tree.predict(X_test)
        
        train_errors.append(mean_squared_error(y_train, current_prediction_train))
        test_errors.append(mean_squared_error(y_test, current_prediction_test))
    
    return train_errors, test_errors



# Параметры
n_trees = 100
max_depth = 3
eta = 0.1
subsample = 0.5  # размер подвыборки для стохастического варианта


# Запуск алгоритмов

train_errors_orig, test_errors_orig = gb_fit_original(
    n_trees, max_depth, X_train, X_test, y_train, y_test, eta
)

train_errors_stoch, test_errors_stoch = gb_fit_stochastic(
    n_trees, max_depth, X_train, X_test, y_train, y_test, eta, subsample
)

# =============================================
# 4. Построение графиков
# =============================================
plt.figure(figsize=(15, 10))

# График 1: Ошибки на тестовой выборке (основной график из задания)
plt.subplot(2, 2, 1)
plt.plot(range(1, n_trees + 1), test_errors_orig, 'b-', linewidth=2, label='Оригинальный GB')
plt.plot(range(1, n_trees + 1), test_errors_stoch, 'r-', linewidth=2, label=f'Стохастический GB (subsample={subsample})')
plt.xlabel('Номер итерации (количество деревьев)')
plt.ylabel('MSE на тестовой выборке')
plt.title('Сравнение ошибки на тестовой выборке')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(2000, 4500)

# Добавляем вертикальные линии для лучших ошибок
best_orig = min(test_errors_orig)
best_stoch = min(test_errors_stoch)
best_orig_iter = test_errors_orig.index(best_orig) + 1
best_stoch_iter = test_errors_stoch.index(best_stoch) + 1

plt.axvline(x=best_orig_iter, color='blue', linestyle='--', alpha=0.5, 
            label=f'Лучший оригинальный: {best_orig:.0f} (итерация {best_orig_iter})')
plt.axvline(x=best_stoch_iter, color='red', linestyle='--', alpha=0.5,
            label=f'Лучший стохастический: {best_stoch:.0f} (итерация {best_stoch_iter})')

plt.legend()

# График 2: Ошибки на тренировочной выборке
plt.subplot(2, 2, 2)
plt.plot(range(1, n_trees + 1), train_errors_orig, 'b-', linewidth=2, alpha=0.7, label='Оригинальный GB (train)')
plt.plot(range(1, n_trees + 1), train_errors_stoch, 'r-', linewidth=2, alpha=0.7, label='Стохастический GB (train)')
plt.plot(range(1, n_trees + 1), test_errors_orig, 'b--', linewidth=1, alpha=0.5, label='Оригинальный GB (test)')
plt.plot(range(1, n_trees + 1), test_errors_stoch, 'r--', linewidth=1, alpha=0.5, label='Стохастический GB (test)')
plt.xlabel('Номер итерации (количество деревьев)')
plt.ylabel('MSE')
plt.title('Ошибки на тренировочной и тестовой выборках')
plt.grid(True, alpha=0.3)
plt.legend()

# График 3: Разница между train и test ошибками
plt.subplot(2, 2, 3)
diff_orig = np.array(test_errors_orig) - np.array(train_errors_orig)
diff_stoch = np.array(test_errors_stoch) - np.array(train_errors_stoch)

plt.plot(range(1, n_trees + 1), diff_orig, 'b-', linewidth=2, label='Оригинальный GB')
plt.plot(range(1, n_trees + 1), diff_stoch, 'r-', linewidth=2, label='Стохастический GB')
plt.xlabel('Номер итерации (количество деревьев)')
plt.ylabel('Test MSE - Train MSE')
plt.title('Степень переобучения (разница ошибок)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.legend()

# График 4: Детальное сравнение лучших ошибок
plt.subplot(2, 2, 4)
iterations = list(range(1, min(51, n_trees) + 1))  # Первые 50 итераций

plt.plot(iterations, test_errors_orig[:len(iterations)], 'b-', linewidth=2, marker='o', markersize=4, label='Оригинальный GB')
plt.plot(iterations, test_errors_stoch[:len(iterations)], 'r-', linewidth=2, marker='s', markersize=4, label='Стохастический GB')

# Отмечаем точки минимумов
plt.scatter([best_orig_iter], [best_orig], color='blue', s=100, zorder=5, 
            edgecolors='black', label=f'Min оригинальный: {best_orig:.0f}')
plt.scatter([best_stoch_iter], [best_stoch], color='red', s=100, zorder=5,
            edgecolors='black', label=f'Min стохастический: {best_stoch:.0f}')

plt.xlabel('Номер итерации (первые 50 деревьев)')
plt.ylabel('MSE на тестовой выборке')
plt.title('Детальное сравнение (первые 50 итераций)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))

subsample_sizes = [0.3, 0.5, 0.7, 1.0]  # 1.0 - это оригинальный GB
colors = ['green', 'red', 'orange', 'blue']
labels = ['subsample=0.3', 'subsample=0.5', 'subsample=0.7', 'subsample=1.0 (оригинальный)']

for subsample_size, color, label in zip(subsample_sizes, colors, labels):
    if subsample_size == 1.0:
        # Используем оригинальный алгоритм
        _, test_errors = gb_fit_original(n_trees, max_depth, X_train, X_test, y_train, y_test, eta)
    else:
        # Используем стохастический алгоритм
        _, test_errors = gb_fit_stochastic(n_trees, max_depth, X_train, X_test, y_train, y_test, eta, subsample_size)
    
    plt.plot(range(1, n_trees + 1), test_errors, color=color, linewidth=2, label=label)
    
    best_error = min(test_errors)
    best_iter = test_errors.index(best_error) + 1
    print(f"{label:20} | Min MSE: {best_error:.0f} на итерации {best_iter}")

plt.xlabel('Номер итерации (количество деревьев)')
plt.ylabel('MSE на тестовой выборке')
plt.title('Влияние размера подвыборки на ошибку')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(2000, 4500)

plt.tight_layout()
plt.show()

