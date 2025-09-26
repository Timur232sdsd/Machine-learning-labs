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
eta = 1e-2
epsilon = 1e-6
max_iter = 10000

W = np.array([1.0, 0.5]) 
print(f'Number of objects = {n} \nLearning rate = {eta} \nInitial weights = {W} \n')

for i in range(max_iter):
    W_old = W.copy()  
    y_pred = np.dot(X, W) 
    gradient = (2/n) * np.dot(X.T, y_pred - y) 
    W -= eta * gradient
    

    weight_change = np.linalg.norm(W - W_old)
    if weight_change < epsilon:
        print(f'Iteration #{i}: W = {W}, изменение весов {weight_change:.6f} < {epsilon}')
        break
        

    if i % 10 == 0:
        err = calc_mse(y, y_pred)
        print(f'Iteration #{i}: W = {W}, MSE = {err:.2f}, изменение весов = {weight_change:.6f}')
else:
    print(f'Достигнуто максимальное количество итераций {max_iter}')