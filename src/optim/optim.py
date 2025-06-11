import numpy as np


class SGD:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, parameters: list[tuple], gradients: list[tuple]) -> list[tuple]:
        results = []
        for params, grads in zip(parameters, gradients):
            res = []
            for param, grad in zip(params, grads):
                res.append(param - self.learning_rate * grad)
            results.append(tuple(res))
        return results
