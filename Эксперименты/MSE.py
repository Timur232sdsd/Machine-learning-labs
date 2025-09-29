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


def gradient_descent_reg_l1(X, y, iterations, eta=1e-4, reg=1e-8):
    W = np.random.randn(X.shape[1])
    n = X.shape[0]
    
    for i in range(iterations):
        y_pred = np.dot(X, W)
        err = calc_mse(y, y_pred)
        
        # Градиент MSE (без изменений)
        dQ = 2/n * X.T @ (y_pred - y)
        
        # Градиент L1-регуляризации (субградиент)
        dReg = reg * np.sign(W)
        
        # Комбинированное обновление
        W -= eta * (dQ + dReg)
        
        if i % (iterations // 10) == 0:
            print(f'Iter: {i}, weights: {W}, error {err}')
    
    print(f'Final MSE: {calc_mse(y, np.dot(X, W))}')
    return W
