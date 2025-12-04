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

# 1. Базовая версия градиентного бустинга
def gb_fit_slow(n_trees, max_depth, X_train, X_test, y_train, y_test, eta):
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
        
        pred_train = tree.predict(X_train)
        pred_test = tree.predict(X_test)
        
        current_prediction_train += eta * pred_train
        current_prediction_test += eta * pred_test
        
        train_errors.append(np.mean((y_train - current_prediction_train) ** 2))
        test_errors.append(np.mean((y_test - current_prediction_test) ** 2))
    
    return train_errors, test_errors

def gb_fit_fast(n_trees, max_depth, X_train, X_test, y_train, y_test, eta):
    train_errors = []
    test_errors = []
    
    current_prediction_train = np.zeros(len(y_train), dtype=np.float32)
    current_prediction_test = np.zeros(len(y_test), dtype=np.float32)
    
    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        if i == 0:
            tree.fit(X_train, y_train)
        else:
            residuals = y_train - current_prediction_train
            tree.fit(X_train, residuals)
        
        pred_train = tree.predict(X_train).astype(np.float32)
        pred_test = tree.predict(X_test).astype(np.float32)
        
        np.add(current_prediction_train, eta * pred_train, out=current_prediction_train)
        np.add(current_prediction_test, eta * pred_test, out=current_prediction_test)
        
        train_errors.append(np.mean((y_train - current_prediction_train) ** 2))
        test_errors.append(np.mean((y_test - current_prediction_test) ** 2))
    
    return train_errors, test_errors

# 3. Оптимизированная версия с ранней остановкой
def gb_fit_fast_early_stop(n_trees, max_depth, X_train, X_test, y_train, y_test, eta, early_stop=10):
    train_errors = []
    test_errors = []
    
    current_prediction_train = np.zeros(len(y_train), dtype=np.float32)
    current_prediction_test = np.zeros(len(y_test), dtype=np.float32)
    
    best_test_error = float('inf')
    no_improvement = 0
    
    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        if i == 0:
            tree.fit(X_train, y_train)
        else:
            residuals = y_train - current_prediction_train
            tree.fit(X_train, residuals)
        
        pred_train = tree.predict(X_train).astype(np.float32)
        pred_test = tree.predict(X_test).astype(np.float32)
        
        np.add(current_prediction_train, eta * pred_train, out=current_prediction_train)
        np.add(current_prediction_test, eta * pred_test, out=current_prediction_test)
        
        train_error = np.mean((y_train - current_prediction_train) ** 2)
        test_error = np.mean((y_test - current_prediction_test) ** 2)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        # Проверка ранней остановки
        if test_error < best_test_error:
            best_test_error = test_error
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement >= early_stop:
            break
    
    return train_errors, test_errors

# Параметры для экспериментов
n_trees = 100
max_depth = 3
eta = 0.1

# Запуск всех версий
train_slow, test_slow = gb_fit_slow(n_trees, max_depth, X_train, X_test, y_train, y_test, eta)
train_fast, test_fast = gb_fit_fast(n_trees, max_depth, X_train, X_test, y_train, y_test, eta)
train_early, test_early = gb_fit_fast_early_stop(n_trees, max_depth, X_train, X_test, y_train, y_test, eta, early_stop=10)

# Построение графиков
plt.figure(figsize=(12, 4))

# График 1: Сравнение ошибок на тестовой выборке
plt.subplot(1, 2, 1)
plt.plot(range(1, len(test_slow) + 1), test_slow, 'b-', label='Базовая', linewidth=2)
plt.plot(range(1, len(test_fast) + 1), test_fast, 'r-', label='Оптимизированная', linewidth=2)
plt.plot(range(1, len(test_early) + 1), test_early, 'g-', label='С ранней остановкой', linewidth=2)
plt.xlabel('Количество деревьев')
plt.ylabel('MSE на тестовой выборке')
plt.title('Сравнение версий градиентного бустинга')
plt.grid(True, alpha=0.3)
plt.legend()

# График 2: Финальные ошибки
plt.subplot(1, 2, 2)
final_errors = [test_slow[-1], test_fast[-1], test_early[-1]]
labels = ['Базовая', 'Оптимизированная', 'Ранняя остановка']
colors = ['blue', 'red', 'green']

bars = plt.bar(labels, final_errors, color=colors, alpha=0.7)
plt.ylabel('Финальная MSE')
plt.title('Финальные ошибки на тестовой выборке')
plt.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bar, error in zip(bars, final_errors):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{error:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Минимальные ошибки для анализа
min_slow = min(test_slow)
min_fast = min(test_fast)
min_early = min(test_early)

# Количество деревьев для ранней остановки
n_trees_early = len(test_early)