import numpy as np

import src.functional.functional as F


class Layer:
    def __call__(self):
        raise NotImplementedError("ERROR: Forward function not implemented in layer!")

    def back(self):
        raise NotImplementedError("ERROR: Backward function not implemented in layer!")


class SiLU(Layer):
    def __init__(self, eval: bool = False):
        self.eval = eval
        
        self.function = (lambda x: x * F.sigmoid(x))
        self.batch = None
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if self.eval:
            self.batch = batch
        return self.function(batch)
    
    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        if self.eval:
            return dloss_dout
        
        sigmoid_x = F.sigmoid(self.batch)
        dfunc_dx = sigmoid_x * (1 + dloss_dout * (1 - sigmoid_x)) 
        
        dloss_dbatch = dloss_dout * dfunc_dx

        return dloss_dbatch


class Softmax(Layer):
    pass
    

class Conv(Layer):
    def __init__(self, Cin: int, Cout: int,
                 kernel_size: tuple[int, int],
                 padding: tuple[int, int] = (0, 0),
                 stride: tuple[int, int] = (1, 1),
                 eval: bool = False):
        # channels in-out
        self.Cin = Cin
        self.Cout = Cout
        # kernel size plt.savefig("sample.png")
        self.Kh = kernel_size[0]
        self.Kw = kernel_size[1]
        # padding
        self.Ph = padding[0]
        self.Pw = padding[1]
        # stride
        self.Sh = stride[0]
        self.Sw = stride[1]
        
        self.eval = eval
        self.batch = None
        self.conv_r = None
        
        self.padding = ((0, 0), (self.Ph, self.Pw), (self.Pw, self.Ph), (0, 0))
        self.kernel = F.he_initialization((self.Kh * self.Kw * self.Cin, self.Cout), self.Kh * self.Kw * self.Cin)
        self.bias = np.zeros((self.Cout,), dtype=np.float64)
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if self.eval:
            self.batch = batch

        padded_batch = np.pad(batch, self.padding)
        Bs, Hin, Win, _ = padded_batch.shape
        Nh = (Hin - self.Kh) // self.Sh + 1
        Nw = (Win - self.Kw) // self.Sw + 1
        
        BatchCols = F.Batch2Col(padded_batch, (self.Kh, self.Kw), (self.Sh, self.Sw))
        BatchColsT = BatchCols.transpose((0, 2, 1))
        
        conv = np.matmul(BatchColsT, self.kernel)
        conv_r = conv.reshape((Bs, Nh, Nw, self.Cout)) + self.bias

        if self.eval:
            self.conv_r
        
        return conv_r
    
    def back(self, dloss_dout: np.ndarray, lr: float) -> np.ndarray:
        if self.eval:
            return dloss_dout

        dloss_dbias = None


class BatchNorm(Layer):
    pass


class MaxPoolling(Layer):
    pass


class Flatten(Layer):
    pass


class Fc(Layer):
    def __init__(self, Cin: int, Cout: int, eval: bool = False):
        self.eval = eval
        self.Cin = Cin
        self.Cout = Cout

        self.W = np.zeros((Cout, Cin), dtype=np.float64)
        self.bias = np.zeros((Cout,), dtype=np.float64)
        
        self.X = None
        self.dW = None
        self.db = None
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        if not self.eval:
            self.X = X
        return np.dot(X, self.W.T) + self.bias
    
    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        if self.eval:
            return dloss_dout

        if not self.eval:
            self.dW = np.dot(dloss_dout.T, self.X)
            self.db = np.sum(dloss_dout, axis=0, keepdims=True).T / dloss_dout.shape[0]

        dX = np.dot(dloss_dout, self.W)
        
        return dX
    
    def update_params(self, Win: np.ndarray, b_in: np.ndarray) -> None:
        self.W = Win
        self.bias = b_in
        
    def zero_grad(self) -> None:
        self.dW = None
        self.db = None
    
    def parameters(self) -> tuple:
        return self.W, self.bias
    
    def gradients(self) -> tuple:
        return self.dW, self.db
