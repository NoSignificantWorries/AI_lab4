import pickle

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


def CNG(w, f, epsilon=1e-6):
    # Compute Numerical Gradient
    grad = np.zeros_like(w)
    it = np.nditer(w, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        ix = it.multi_index
        old_value = w[ix]

        w[ix] = old_value + epsilon
        pos = f(w)

        w[ix] = old_value - epsilon
        neg = f(w)

        w[ix] = old_value

        grad[ix] = (pos - neg) / (2 * epsilon)
        it.iternext()
    
    return grad


def unpickle(filepath: str) -> dict:
    with open(filepath, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return meta


def BatchOHE(data: np.ndarray, batch_size: int, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float64)
    one_hot[np.arange(batch_size), data] = 1

    return one_hot


def OHE_to_indeces(one_hot: np.ndarray) -> np.ndarray:
    return np.argmax(one_hot, axis=-1)


def calculate_TP_TN_FP_FN(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> list:
    metrics = []
    for i in range(num_classes):
        TP = np.sum((y_true == i) & (y_pred == i))
        TN = np.sum((y_true != i) & (y_pred != i))
        FP = np.sum((y_true != i) & (y_pred == i))
        FN = np.sum((y_true == i) & (y_pred != i))
        metrics.append((int(TP), int(TN), int(FP), int(FN)))
    return metrics


def calculate_metrics(TP: int, TN: int, FP: int, FN: int) -> tuple[int, int, int, int]:
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    return precision, recall, f1, accuracy


def calculate_running_metrics(running_metrics, y_true, y_pred, num_classes):
    metrics = calculate_TP_TN_FP_FN(y_true, y_pred, num_classes)
    for i, (TP, TN, FP, FN) in enumerate(metrics):
        running_metrics[i][0] += TP
        running_metrics[i][1] += TN
        running_metrics[i][2] += FP
        running_metrics[i][3] += FN

    return running_metrics


def calculate_aggregated_metrics(metrics: list[tuple[int, int, int, int]]) -> tuple[float, float, float]:
    num_classes = len(metrics)

    # Macro-average
    macro_precision = np.mean([calculate_metrics(*metrics[i])[0] for i in range(num_classes)])
    macro_recall = np.mean([calculate_metrics(*metrics[i])[1] for i in range(num_classes)])
    macro_f1 = np.mean([calculate_metrics(*metrics[i])[2] for i in range(num_classes)])
    
    # Micro-average
    total_tp = sum([metrics[i][0] for i in range(num_classes)])
    total_fp = sum([metrics[i][2] for i in range(num_classes)])
    total_fn = sum([metrics[i][3] for i in range(num_classes)])
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Weighted-average
    '''
    weighted_precision = np.sum([calculate_metrics(*metrics[i])[0] * num_samples_per_class[i] for i in range(num_classes)]) / np.sum(num_samples_per_class)
    weighted_recall = np.sum([calculate_metrics(*metrics[i])[1] * num_samples_per_class[i] for i in range(num_classes)]) / np.sum(num_samples_per_class)
    weighted_f1 = np.sum([calculate_metrics(*metrics[i])[2] * num_samples_per_class[i] for i in range(num_classes)]) / np.sum(num_samples_per_class)
    '''
    
    return macro_precision, macro_recall, macro_f1
