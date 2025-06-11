import numpy as np


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.sum(loss)


def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return (y_pred - y_true)
  

def BCE(y_true, y_pred):
    return binary_cross_entropy(y_true, y_pred), binary_cross_entropy_derivative(y_true, y_pred)
