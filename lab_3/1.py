import numpy as np

# Все решения есть в ex_3/ex_3.pynb

def calc_logloss_hw(y, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    err = -np.mean(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))
    return err


def calc_logloss(y, y_pred):
    err = - np.mean(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))
    return err

