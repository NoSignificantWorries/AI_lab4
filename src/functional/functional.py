import numpy as np


def Batch2Col(batch: np.ndarray,
              kernel_shape: tuple[int, int],
              stride: tuple[int, int] = (1, 1)) -> np.ndarray:
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


def Col2Batch(col: np.ndarray,
              input_shape: tuple[int, int, int, int],
              kernel_shape: tuple[int, int],
              stride: tuple[int, int] = (1, 1)) -> np.ndarray:

    Bs, Hin, Win, Cin = input_shape
    Kh, Kw = kernel_shape
    Sh, Sw = stride

    Nh = (Hin - Kh) // Sh + 1
    Nw = (Win - Kw) // Sw + 1

    patches = col.reshape(Bs, Kh, Kw, Cin, Nh, Nw)
    patches = patches.transpose(0, 4, 5, 1, 2, 3)  # (Bs, Nh, Nw, Kh, Kw, Cin)

    batch = np.zeros(input_shape, dtype=col.dtype)

    for b in range(Bs):
        for h in range(Nh):
            for w in range(Nw):
                h_start = h * Sh
                w_start = w * Sw
                batch[b, h_start:h_start+Kh, w_start:w_start+Kw, :] += patches[b, h, w, :, :, :]

    return batch


def he_initialization(shape: tuple, n_in: int) -> np.ndarray:
    std = np.sqrt(2.0 / n_in)
    W = np.random.randn(*shape) * std
    return W


def sigmoid(data: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-data))


def softmax(data: np.ndarray) -> np.ndarray:
    exp_data = np.exp(data - np.max(data, axis=-1, keepdims=True))
    return exp_data / np.sum(exp_data, axis=-1, keepdims=True)
