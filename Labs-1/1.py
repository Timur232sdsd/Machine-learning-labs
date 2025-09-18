import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err

y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60]

X = np.array([[ 1,  1],
              [ 1,  1],
              [ 1,  2],
              [ 1,  5],
              [ 1,  3],
              [ 1,  0],
              [ 1,  5],
              [ 1, 10],
              [ 1,  1],
              [ 1,  2]])


n = X.shape[0]
eta = 0.1  # Learning rate можно увеличить после нормализации
n_iter = 1000
W = np.array([1.0, 0.5])  # Инициализируем веса

errors = []  # Для сохранения ошибок на каждой итерации

print(f'Number of objects = {n} \nLearning rate = {eta} \nInitial weights = {W} \n')

for i in range(n_iter):
    y_pred = np.dot(X, W)
    err = calc_mse(y, y_pred)
    errors.append(err)
    
    # Обновляем веса
    grad = (2/n) * np.dot(X.T, (y_pred - y))
    W -= eta * grad
    
    if i % 100 == 0:
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 4)}')

# Вычисляем аналитическое решение для нормализованных данных
X_with_intercept = np.column_stack([np.ones(n), X])
theta_analytical = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
print("Аналитическое решение (МНК):", theta_analytical)

# Построение графика
plt.plot(errors)
plt.xlabel('Итерация')
plt.ylabel('MSE')
plt.title('График изменения ошибки')
plt.show()



n = X.shape[0]

eta = 0.099
n_iter = 1000

# W = np.array([1, 0.5])
# print(f'Number of objects = {n} \
#        \nLearning rate = {eta} \
#        \nInitial weights = {W} \n')

# for i in range(n_iter):
#     y_pred = np.dot(X, W)
#     err = calc_mse(y, y_pred)
#     # for k in range(W.shape[0]):
#     #     W[k] -= eta * (1/n * 2 * X[:, k] @ (y_pred - y))
#     W -= eta * (1/n * 2 * np.dot(X.T, y_pred - y))

#     if i % 10 == 0:
#         eta /= 1.1
#         print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)}')
# th = np.linalg.inv(X.T @ X) @ X.T @ y
# print("Аналитическое решение (МНК):", th)
# plt.plot(errors)
# plt.xlabel('Итерация')
# plt.ylabel('MSE')
# plt.title('График изменения ошибки')
# plt.show()