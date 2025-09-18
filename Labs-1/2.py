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


#Нерабочий код 
# n = X.shape[0]

# eta = 1e-2 
# n_iter = 100

# W = np.array([1, 0.5])
# print(f'Number of objects = {n} \
#        \nLearning rate = {eta} \
#        \nInitial weights = {W} \n')

# for i in range(n_iter):
#     y_pred = np.dot(X, W)
#     err = calc_mse(y, y_pred)

#     W -= eta * (1/n * 2 * np.dot(X.T, y_pred - y))

#     if i % 10 == 0:
#         print(f'Iteration #{i}: W_new = {W}, MSE = {round(err,2)}')


#Рабочий

n = X.shape[0]

eta = 1e-2 
n_iter = 100

W = np.array([1, 0.5])
print(f'Number of objects = {n} \
       \nLearning rate = {eta} \
       \nInitial weights = {W} \n')

for i in range(n_iter):
    y_pred = np.dot(X, W)
    err = calc_mse(y, y_pred)

    W -= eta * (1/n * 2 * np.dot(X.T, y_pred - y))

    if i % 10 == 0:
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err,2)}')
