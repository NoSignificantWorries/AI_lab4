import pickle

import numpy as np

import src.nodes.nodes as nn


class BaseModel:
    def forward(self):
        raise NotImplementedError("ERROR: Forward not implemented!")
    
    def backward(self):
        raise NotImplementedError("ERROR: Backward not implemented!")

    def parameters(self):
        raise NotImplementedError("ERROR: parameters not implemented!")
    
    def gradients(self):
        raise NotImplementedError("ERROR: gradients not implemented!")
        
    def update_params(self):
        raise NotImplementedError("ERROR: update_params not implemented!")
        
    def zero_grad(self):
        raise NotImplementedError("ERROR: zero_grad not implemented!")
    
    def save(self):
        raise NotImplementedError("ERROR: save not implemented")
    
    def load(self):
        raise NotImplementedError("ERROR: load not implemented")


class XOR(BaseModel):
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


class Conv(BaseModel):
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


class CNNSmall(BaseModel):
    def __init__(self, eval: bool = False):
        self.eval = eval

        self.cv1 = nn.Conv(3, 32, (3, 3), (1, 1), eval=eval)
        self.bn1 = nn.BatchNorm(32, eval=eval)
        self.SiLU1 = nn.SiLU(eval=eval)
        self.mp1 = nn.MaxPoolling((2, 2), (0, 0), (2, 2), eval=eval)
        self.cv2 = nn.Conv(32, 64, (3, 3), (1, 1), eval=eval)
        self.SiLU2 = nn.SiLU(eval=eval)
        self.bn2 = nn.BatchNorm(64, eval=eval)
        self.mp2 = nn.MaxPoolling((2, 2), (0, 0), (1, 1))
        self.flatten = nn.Flatten(eval=eval)
        self.fc1 = nn.Fc(14400, 128, eval=eval)
        self.SiLU3 = nn.SiLU(eval=eval)
        self.fc2 = nn.Fc(128, 10, eval=eval)
        self.softmax = nn.Softmax(eval=eval)
    
    def forward(self, batch: np.ndarray) -> np.ndarray:
        x = self.cv1(batch)
        x = self.bn1(x)
        x = self.SiLU1(x)
        x = self.mp1(x)
        x = self.cv2(x)
        x = self.SiLU2(x)
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.SiLU3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def backward(self, dloss: np.ndarray) -> np.ndarray:
        d = self.softmax.back(dloss)
        d = self.fc2.back(d)
        d = self.SiLU3.back(d)
        d = self.fc1.back(d)
        d = self.flatten.back(d)
        d = self.mp2.back(d)
        d = self.bn2.back(d)
        d = self.SiLU2.back(d)
        d = self.cv2.back(d)
        d = self.mp1.back(d)
        d = self.bn1.back(d)
        d = self.SiLU1.back(d)
        d = self.cv1.back(d)
        return d
    
    def parameters(self) -> list[tuple]:
        return [self.cv1.parameters(),
                self.bn1.parameters(),
                self.cv2.parameters(),
                self.bn2.parameters(),
                self.fc1.parameters(),
                self.fc2.parameters()]
    
    def gradients(self) -> list[tuple]:
        return [self.cv1.gradients(),
                self.bn1.gradients(),
                self.cv2.gradients(),
                self.bn2.gradients(),
                self.fc1.gradients(),
                self.fc2.gradients()]
        
    def update_params(self,
                      cv1_p: tuple,
                      bn1_p: tuple,
                      cv2_p: tuple,
                      bn2_p: tuple,
                      fc1_p: tuple,
                      fc2_p: tuple) -> None:
        self.cv1.update_params(*cv1_p)
        self.bn1.update_params(*bn1_p)
        self.cv2.update_params(*cv2_p)
        self.bn2.update_params(*bn2_p)
        self.fc1.update_params(*fc1_p)
        self.fc2.update_params(*fc2_p)
        
    def zero_grad(self) -> None:
        self.cv1.zero_grad()
        self.bn1.zero_grad()
        self.cv2.zero_grad()
        self.bn2.zero_grad()
        self.fc1.zero_grad()
        self.fc2.zero_grad()
    
    def save(self, filepath: str) -> None:
        params = {
            b"cv1": self.cv1.parameters(),
            b"bn1": self.bn1.parameters(),
            b"cv2": self.cv2.parameters(),
            b"bn2": self.bn2.parameters(),
            b"fc1": self.fc1.parameters(),
            b"fc2": self.fc2.parameters()
        }
        with open(filepath, "wb") as f:
            pickle.dump(params, f)
    
    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            params = pickle.load(f)
        self.cv1.update_params(*params[b"cv1"])
        self.bn1.update_params(*params[b"bn1"])
        self.cv2.update_params(*params[b"cv2"])
        self.bn2.update_params(*params[b"bn2"])
        self.fc1.update_params(*params[b"fc1"])
        self.fc2.update_params(*params[b"fc2"])
