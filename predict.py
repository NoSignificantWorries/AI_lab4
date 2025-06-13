import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import src.models.models as M
import src.functional.functional as F
import src.dataflow.dataflow as data


def main(filepath: str, weights_path: str) -> None:
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape

    if h != 32 or w != 32:
        print("WARNING: Image dimensions are not 32x32. Image will be resize to 32x32.")
        img = cv2.resize(img, (32, 32))
    
    img = np.array([img], dtype=np.float64)
    img = img / 255
    

    model = M.CNNSmall(eval=True)
    model.load(weights_path)
    
    print("Class:", data.CLASS_NAMES[F.OHE_to_indeces(model.forward(img))[0]])


if __name__ == "__main__":
    main("cat.jpg", "results/model_last.plk")
