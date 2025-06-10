import numpy as np


def mse_loss(Y_hat, Y):
    m = Y.shape[1]
    loss = np.sum((Y_hat - Y) ** 2) / (2 * m)
    dA = (Y_hat - Y) / m
    return loss, dA
