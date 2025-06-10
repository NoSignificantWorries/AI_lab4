import numpy as np


class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for W, b in self.parameters:
            W -= self.learning_rate * W.grad
            b -= self.learning_rate * b.grad

    def zero_grad(self):
        for W, b in self.parameters:
            W.grad = None
            b.grad = None
