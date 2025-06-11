import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import src.nodes.nodes as nn
import src.functional.functional as F


def unpickle(filepath: str) -> dict:
    with open(filepath, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return meta


class Model:
    def __init__(self, eval: bool = False):
        self.conv1 = nn.Conv(3, 32, (3, 3), (1, 1), eval=eval)
        self.SiLU1 = nn.SiLU(eval=eval)
    
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        x = self.conv1(batch)
        x = self.SiLU1(x)
        return x
    
    def back(self):
        pass


def main() -> None:
    num_classes = 10
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    test_batch = unpickle("data/test_batch")

    test_batch_labels = test_batch[b"labels"]
    test_batch_data = test_batch[b"data"].reshape((-1, 3, 32, 32))
    test_batch_data = test_batch_data.transpose((0, 2, 3, 1))
    test_batch_data = test_batch_data / 255
    print(test_batch_data.shape)
    
    batch = test_batch_data[:2]
    
    batch = batch - batch.min()
    batch = batch / batch.max()
    
    pool = nn.MaxPoolling((2, 2), (0, 0), (2, 2))
    pooled_batch = pool(batch)
    back_batch = pool.back(pooled_batch)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    
    ax[0][0].imshow(test_batch_data[0])
    ax[0][1].imshow(test_batch_data[1])
    ax[1][0].imshow(back_batch[0])
    ax[1][1].imshow(back_batch[1])

    plt.savefig("sample.png")


if __name__ == "__main__":
    main()
