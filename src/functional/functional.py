import numpy as np


def Batch2Col(batch: np.ndarray, kernel_shape: tuple[int, int], stride: tuple[int, int] = (1, 1)) -> np.ndarray:
    Bs, Hin, Win, Cin = batch.shape
    Kh, Kw = kernel_shape
    Sh, Sw = stride
    
    Nh = (Hin - Kh) // Sh + 1
    Nw = (Win - Kw) // Sw + 1
    
    patches = np.lib.stride_tricks.as_strided(
        batch,
        shape=(Bs, Nh, Nw, Kh, Kw, Cin),
        strides=(batch.strides[0], Sh * batch.strides[1], Sw * batch.strides[2], batch.strides[1], batch.strides[2], batch.strides[3])
    )
    # (Bs, Kh, Kw, Cin, Nh, Nw)
    patches = patches.transpose(0, 3, 4, 5, 1, 2)
    new_batch = patches.reshape(Bs, Kh * Kw * Cin, Nh * Nw)

    return new_batch


def he_initialization(shape: tuple, n_in: int) -> np.ndarray:
    std = np.sqrt(2.0 / n_in)
    W = np.random.randn(*shape) * std
    return W


def sigmoid(data: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-data))
