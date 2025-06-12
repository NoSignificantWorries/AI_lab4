import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import src.loss.loss as criterion
import src.optim.optim as optim
import src.nodes.nodes as nn
import src.functional.functional as F
import src.models.models as M


def unpickle(filepath: str) -> dict:
    with open(filepath, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return meta


def main() -> None:
    num_classes = 10
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    test_batch = unpickle("data/test_batch")

    test_batch_labels = np.array(test_batch[b"labels"])
    test_batch_data = test_batch[b"data"].reshape((-1, 3, 32, 32))
    test_batch_data = test_batch_data.transpose((0, 2, 3, 1))
    test_batch_data = test_batch_data / 255
    print(test_batch_data.shape)
    
    batch = test_batch_data[:5]
    labels = test_batch_labels[:5]
    print(labels)
    one_hot = np.zeros((5, num_classes), dtype=np.float64)
    one_hot[np.arange(5), labels.astype(np.uint32)] = 1
    print(one_hot)
    
    batch = batch - batch.min()
    batch = batch / batch.max()
    
    model = M.CNNSmall()
    optimizer = optim.SGD(0.001)

    for _ in range(100):
        pred = model.forward(batch)
        print(np.round(pred))
        loss, dloss = criterion.CEL(one_hot, pred)
        print(loss)
        model.backward(dloss)
        
        new_param = optimizer.step(model.parameters(), model.gradients())
        
        model.update_params(*new_param)
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    
    ax[0][0].imshow(test_batch_data[0])
    ax[0][1].imshow(test_batch_data[1])
    # ax[1][0].imshow(back_batch[0])
    # ax[1][1].imshow(back_batch[1])

    plt.savefig("sample.png")


if __name__ == "__main__":
    main()
