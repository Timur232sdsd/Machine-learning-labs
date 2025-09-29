import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X, Y, coef = datasets.make_regression(n_samples=1000, n_features=2, n_informative=2, 
                                      n_targets=1, noise=5, coef=True, random_state=2)

X[:, 0] *= 10

# Нормализация
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
X = (X - means) / stds  # Векторизованная нормализация вместо циклов


X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

def calc_mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def gradient_descent(X, y, iterations, eta):
    W = np.random.randn(X.shape[1])
    errors = []
    
    for i in range(iterations):
        y_pred = np.dot(X, W)
        dQ = 2 / len(y) * X.T @ (y_pred - y)
        W -= eta * dQ
        
        if i % 10 == 0:
            error = calc_mse(y, np.dot(X, W))
            errors.append(error)
            
    return W, errors

def stohastic_gradient_descent(X, y, iterations, size, eta):
    W = np.random.randn(X.shape[1])
    errors = []
    n = X.shape[0]
    
    for i in range(iterations):
        inds = np.random.choice(n, size=size, replace=False)
        
        X_tmp = X[inds]
        y_tmp = y[inds]
        y_pred_tmp = np.dot(X_tmp, W)
        
        dQ = 2 / size * X_tmp.T @ (y_pred_tmp - y_tmp)
        W -= eta * dQ
        
        if i % 10 == 0:
            error = calc_mse(y, np.dot(X, W))
            errors.append(error)
            
    return W, errors

iterations = 1000
eta = 0.01
batch_size = 32

W_gd, errors_gd = gradient_descent(X_with_bias, Y, iterations, eta)
W_sgd, errors_sgd = stohastic_gradient_descent(X_with_bias, Y, iterations, batch_size, eta)

# Визуализация
plt.figure(figsize=(12, 6))

# График ошибок
plt.subplot(1, 2, 1)
plt.plot(errors_gd, label='GD', color='blue', alpha=0.7)
plt.plot(errors_sgd, label='SGD', color='red', alpha=0.7)
plt.xlabel('Итерации (×10)')
plt.ylabel('MSE')
plt.title('Сравнение скорости сходимости GD и SGD')
plt.legend()
plt.grid(True)

# График в логарифмической шкале для лучшей визуализации
plt.subplot(1, 2, 2)
plt.semilogy(errors_gd, label='GD', color='blue', alpha=0.7)
plt.semilogy(errors_sgd, label='SGD', color='red', alpha=0.7)
plt.xlabel('Итерации (×10)')
plt.ylabel('MSE (log scale)')
plt.title('MSE в логарифмической шкале')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод результатов
print(f"Исходные коэффициенты: {coef}")
print(f"GD финальные веса: {W_gd[1:]}")  # Исключаем bias
print(f"SGD финальные веса: {W_sgd[1:]}")
print(f"GD финальная MSE: {errors_gd[-1]:.6f}")
print(f"SGD финальная MSE: {errors_sgd[-1]:.6f}")