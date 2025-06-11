import numpy as np

import src.nodes.nodes as nn


class XOR:
    def __init__(self, eval: bool = False):
        self.eval = eval

        self.fc1 = nn.Fc(2, 4, self.eval)
        self.SiLU1 = nn.SiLU(self.eval)
        self.fc2 = nn.Fc(4, 1, self.eval)
        self.sigmoid = nn.Sigmoid(self.eval)
    
    def forward(self, batch: np.ndarray) -> np.ndarray:
        x = self.fc1(batch)
        x = self.SiLU1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def backward(self, dloss: np.ndarray) -> np.ndarray:
        d = self.sigmoid.back(dloss)
        d = self.fc2.back(d)
        d = self.SiLU1.back(d)
        d = self.fc1.back(d)
        return d
    
    def parameters(self) -> list[tuple]:
        return [self.fc1.parameters(),
                self.fc2.parameters()]
    
    def gradients(self) -> list[tuple]:
        return [self.fc1.gradients(),
                self.fc2.gradients()]
        
    def update_params(self, fc1_p: tuple, fc2_p: tuple) -> None:
        self.fc1.update_params(*fc1_p)
        self.fc2.update_params(*fc2_p)
        
    def zero_grad(self) -> None:
        self.fc1.zero_grad()
        self.fc2.zero_grad()


class Conv:
    def __init__(self, eval: bool = False):
        self.eval = eval

        self.cv1 = nn.Conv(3, 32, (3, 3), (1, 1), eval=eval)
        self.SiLU1 = nn.SiLU(eval=eval)
        self.cv2 = nn.Conv(32, 32, (3, 3), (1, 1), eval=eval)
        self.SiLU2 = nn.SiLU(eval=eval)
        self.mp1 = nn.MaxPoolling((2, 2))
        self.flatten = nn.Flatten(eval=eval)
    
    def forward(self, batch: np.ndarray) -> np.ndarray:
        x = self.cv1(batch)
        x = self.SiLU1(x)
        x = self.cv2(x)
        x = self.SiLU2(x)
        x = self.mp1(x)
        x = self.flatten(x)
        return x
    
    def backward(self, dloss: np.ndarray) -> np.ndarray:
        d = self.flatten.back(dloss)
        d = self.mp1.back(d)
        d = self.SiLU2.back(d)
        d = self.cv2.back(d)
        d = self.SiLU1.back(d)
        d = self.cv1.back(d)
        return d
    
    def parameters(self) -> list[tuple]:
        return [self.cv1.parameters(),
                self.cv2.parameters()]
    
    def gradients(self) -> list[tuple]:
        return [self.cv1.gradients(),
                self.cv2.gradients()]
        
    def update_params(self, cv1_p: tuple, cv2_p: tuple) -> None:
        self.cv1.update_params(*cv1_p)
        self.cv2.update_params(*cv2_p)
        
    def zero_grad(self) -> None:
        self.cv1.zero_grad()
        self.cv2.zero_grad()
