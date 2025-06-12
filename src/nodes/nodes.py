import numpy as np

import src.functional.functional as F


class Layer:
    def __call__(self):
        raise NotImplementedError("ERROR: Forward function not implemented in layer!")

    def back(self):
        raise NotImplementedError("ERROR: Backward function not implemented in layer!")


class Sigmoid(Layer):
    def __init__(self, eval: bool = False):
        self.eval = eval

        self.function = F.sigmoid
        self.batch = None

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if not self.eval:
            self.batch = batch
        return self.function(batch)
    
    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        if self.eval:
            return dloss_dout
        
        sigmoid_x = self.function(self.batch)
        dfunc_dx = sigmoid_x * (1 - sigmoid_x) 
        
        dloss_dbatch = dloss_dout * dfunc_dx

        return dloss_dbatch


class SiLU(Layer):
    def __init__(self, eval: bool = False):
        self.eval = eval
        
        self.function = (lambda x: x * F.sigmoid(x))
        self.batch = None
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if not self.eval:
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
    def __init__(self, eval: bool = False):
        self.eval = eval
        
        self.function = (lambda x: np.apply_along_axis(F.softmax, -1, x))
        self.batch = None
        self.p = None
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if not self.eval:
            self.batch = batch
            self.p = self.function(batch)
            return self.p
        return self.function(batch)
    
    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        if self.eval:
            return dloss_dout

        p_tensor = self.p[:, None] * np.eye(self.p.shape[1])
        jacobian = p_tensor - self.p[:, :, None] * self.p[:, None, :]
        dloss_dbatch = np.einsum('bi,bij->bj', dloss_dout, jacobian)
        
        return dloss_dbatch
    

class Conv(Layer):
    def __init__(self, Cin: int, Cout: int,
                 kernel_size: tuple[int, int],
                 padding: tuple[int, int] = (0, 0),
                 stride: tuple[int, int] = (1, 1),
                 eval: bool = False):

        self.eval = eval

        self.Cin = Cin
        self.Cout = Cout

        self.Kh, self.Kw = kernel_size
        self.Ph, self.Pw = padding
        self.Sh, self.Sw = stride
        
        self.batch_shape = None
        self.BatchCols = None
        
        self.padding = ((0, 0), (self.Ph, self.Pw), (self.Pw, self.Ph), (0, 0))
        self.kernel = F.he_initialization((self.Kh * self.Kw * self.Cin, self.Cout), self.Kh * self.Kw * self.Cin)
        self.bias = np.zeros((self.Cout,), dtype=np.float64)
        
        self.dk = None
        self.db = None
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        padded_batch = np.pad(batch, self.padding)

        self.batch_shape = padded_batch.shape

        Bs, Hin, Win, _ = padded_batch.shape
        Nh = (Hin - self.Kh) // self.Sh + 1
        Nw = (Win - self.Kw) // self.Sw + 1
        
        self.BatchCols = F.Batch2Col(padded_batch, (self.Kh, self.Kw), (self.Sh, self.Sw))
        BatchColsT = self.BatchCols.transpose((0, 2, 1))
        
        conv = np.matmul(BatchColsT, self.kernel)
        conv_r = conv.reshape((Bs, Nh, Nw, self.Cout)) + self.bias

        return conv_r
    
    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        if self.eval:
            return dloss_dout

        self.db = np.sum(dloss_dout, axis=(0, 1, 2))
        
        Bs, Nh, Nw, Cout = dloss_dout.shape
        dloss_dout_r = dloss_dout.reshape((Bs, Nh * Nw, Cout))
        
        self.dK = self.BatchCols @ dloss_dout_r
        self.dK = np.sum(self.dK, axis=0)

        dBatchColsT = np.dot(dloss_dout_r, self.kernel.T)
        dBatchCols = dBatchColsT.transpose((0, 2, 1))
        
        dpadded_batch = F.Col2Batch(dBatchCols, self.batch_shape, (self.Kh, self.Kw), (self.Sh, self.Sw))
        dbatch = dpadded_batch[:, self.Ph:self.batch_shape[1] - self.Ph, self.Pw:self.batch_shape[2] - self.Pw, :]

        return dbatch

    def update_params(self, Kin: np.ndarray, b_in: np.ndarray) -> None:
        self.kernel = Kin
        self.bias = b_in
        
    def zero_grad(self) -> None:
        self.dK = None
        self.db = None
    
    def parameters(self) -> tuple:
        return self.kernel, self.bias
    
    def gradients(self) -> tuple:
        return self.dK, self.db


class BatchNorm(Layer):
    def __init__(self, num_features: int, momentum: float = 0.9, epsilon: float = 1e-15, eval: bool = False):
        self.eval = False
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.cache = None
        self.dgamma = None
        self.dbeta = None
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if not self.eval:
            mean = np.mean(batch, axis=(0, 1, 2))
            var = np.var(batch, axis=(0, 1, 2))

            x_hat = (batch - mean[None, None, None, :]) / np.sqrt(var[None, None, None, :] + self.epsilon)

            out = self.gamma[None, None, None, :] * x_hat + self.beta[None, None, None, :]

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.cache = (batch, mean, var, x_hat)
        else:
            x_hat = (batch - self.running_mean[None, None, None, :]) / np.sqrt(self.running_var[None, None, None, :] + self.epsilon)
            out = self.gamma[None, None, None, :] * x_hat + self.beta[None, None, None, :]

        return out
    
    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        x, mean, var, x_hat = self.cache
        batch_size = np.prod(x.shape[:-1])

        self.dgamma = np.sum(dloss_dout * x_hat, axis=(0, 1, 2))  # Суммируем все, кроме Cin
        self.dbeta = np.sum(dloss_dout, axis=(0, 1, 2))      # Суммируем все, кроме Cin

        dx_hat = dloss_dout * self.gamma[None, None, None, :] #Добавили broadcasting
        dvar = np.sum(dx_hat * (x - mean[None, None, None, :]) * (-0.5) * (var[None, None, None, :] + self.epsilon)**(-1.5), axis=(0, 1, 2))
        dmean = np.sum(dx_hat * (-1) / np.sqrt(var[None, None, None, :] + self.epsilon), axis=(0, 1, 2)) + \
                dvar * np.sum(-2 * (x - mean[None, None, None, :]), axis=(0, 1, 2)) / batch_size

        dx = dx_hat / np.sqrt(var[None, None, None, :] + self.epsilon) + dvar * 2 * (x - mean[None, None, None, :]) / batch_size + dmean / batch_size

        return dx

    def update_params(self, gamma_in: np.ndarray, beta_in: np.ndarray) -> None:
        self.gamma = gamma_in
        self.beta = beta_in
        
    def zero_grad(self) -> None:
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
    
    def parameters(self) -> tuple:
        return self.gamma, self.beta
    
    def gradients(self) -> tuple:
        return self.dgamma, self.dbeta


class MaxPoolling(Layer):
    def __init__(self, pool_size: tuple[int, int], padding: tuple[int, int] = (0, 0), stride: tuple[int, int] = (1, 1), eval: bool = False):
        self.eval = eval

        self.Kh, self.Kw = pool_size
        self.Ph, self.Pw = padding
        self.Sh, self.Sw = stride
        
        self.input_size = None
        self.argmax_indices = None
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        Bs, Hin, Win, Cin = batch.shape
        self.input_size = batch.shape

        windows = np.lib.stride_tricks.sliding_window_view(batch, window_shape=(self.Kh, self.Kw), axis=(1, 2))
        windows = windows[:, ::self.Sh, ::self.Sw, :, :, :]

        Nh = (Hin - self.Kh) // self.Sh + 1
        Nw = (Win - self.Kw) // self.Sw + 1
        windows_reshaped = windows.reshape(Bs, Nh, Nw, Cin, self.Kh * self.Kw)
        
        self.argmax_indices = np.argmax(windows_reshaped, axis=4)
        
        output = np.take_along_axis(windows_reshaped, self.argmax_indices[..., None], axis=4)
        output = output.squeeze(axis=4)
        
        return output
    
    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        Bs, Hin, Win, Cin = self.input_size

        Nh = (Hin - self.Kh) // self.Sh + 1
        Nw = (Win - self.Kw) // self.Sw + 1

        h_coords, w_coords = np.meshgrid(np.arange(Nh), np.arange(Nw), indexing='ij')
        h_coords = h_coords.reshape(-1)
        w_coords = w_coords.reshape(-1)

        h_start = h_coords * self.Sh
        w_start = w_coords * self.Sw

        max_indices_flat = self.argmax_indices.reshape(Bs, Nh * Nw, Cin)

        h_offset = max_indices_flat // self.Kw
        w_offset = max_indices_flat % self.Kw

        h_indices = h_start[None, :, None] + h_offset
        w_indices = w_start[None, :, None] + w_offset

        mask = np.zeros((Bs, Hin, Win, Cin), dtype=dloss_dout.dtype)
        for b in range(Bs):
            for c in range(Cin):
                mask[b, h_indices[b, :, c], w_indices[b, :, c], c] = dloss_dout[b].reshape(Nh*Nw, Cin)[ :, c]

        return mask


class Flatten(Layer):
    def __init__(self, eval: bool = False):
        self.eval = eval
        self.input_shape = None
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if not self.eval:
            self.input_shape = batch.shape
        Bs = batch.shape[0]
        return batch.reshape(Bs, -1)

    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        if self.eval:
            return dloss_dout
        
        return dloss_dout.reshape(self.input_shape)


class Fc(Layer):
    def __init__(self, Cin: int, Cout: int, eval: bool = False):
        self.eval = eval
        self.Cin = Cin
        self.Cout = Cout

        self.W = F.he_initialization((Cout, Cin), Cin)
        self.bias = np.zeros((Cout,), dtype=np.float64)
        
        self.X = None
        self.dW = None
        self.db = None
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        if not self.eval:
            self.X = X
        res = np.dot(self.W, X.T) + self.bias[:, None]
        return res.T
    
    def back(self, dloss_dout: np.ndarray) -> np.ndarray:
        if self.eval:
            return dloss_dout

        if not self.eval:
            self.dW = np.dot(dloss_dout.T, self.X)
            self.db = np.sum(dloss_dout, axis=0).T

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
