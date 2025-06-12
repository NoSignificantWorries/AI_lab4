import numpy as np


def BCE(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    gradient = (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    return loss, gradient


def CEL(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    gradient =  (y_pred - y_true) / y_true.shape[0]
    
    return loss, gradient


def MSE(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = np.mean((y_true - y_pred) ** 2)
    gradient = 2 * (y_pred - y_true) / y_true.size
    
    return loss, gradient
