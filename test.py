import numpy as np

import src.functional.functional as F
import src.nodes.nodes as nn
import src.models.models as m
import src.loss.loss as loss
import src.optim.optim as optim


if __name__ == "__main__":
    XOR_Batch = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    XOR_Batch = XOR_Batch.T
    XOR_Lables = np.array([[0], [1], [1], [0]], dtype=np.float64)
    
    model = m.XOR()
    optimizer = optim.SGD(0.01)
    
    '''
    for _ in range(500):
        pred = model.forward(XOR_Batch)
        pred = np.round(pred)

        print(pred)
        
        l_loss, dL = loss.BCE(XOR_Lables, pred)
        print(l_loss)

        model.backward(dL)

        new_params = optimizer.step(model.parameters(), model.gradients())
        
        model.update_params(*new_params)
        model.zero_grad()    
    '''

    tr = np.array([[[[1, 2, 3]] * 4] * 4], dtype=np.float64)
    print(tr.shape)

    model = m.Conv()

    pred = model.forward(tr)
    print("pred:", pred.shape)
    print(model.backward(pred).shape)
