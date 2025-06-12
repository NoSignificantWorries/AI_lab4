import os
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.loss.loss as criterion
import src.optim.optim as optim
import src.functional.functional as F
import src.models.models as M
import src.dataflow.dataflow as data


def unpickle_all(files_dir_path: str, files: list[str]) -> tuple:
    data = None
    labels = []
    for file in files:
        file_data = F.unpickle(os.path.join(files_dir_path, file))
        if data is None:
            data = file_data[b"data"]
        else:
            data = np.concatenate([data, file_data[b"data"]])
        labels += file_data[b"labels"]

    data = data.reshape((-1, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))
    data = data / 255
    return data, np.array(labels, dtype=np.uint32)


def train(model: M.BaseModel, criterion, optimizer, dataloaders: dict, num_epochs: int, save_path: str, save_period: int = -1) -> None:
    save_path = os.path.join(save_path, datetime.now().strftime("%d%m%Y_%H%M"))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, "weights")):
        os.makedirs(os.path.join(save_path, "weights"))

    results = {
        "train": {
            "loss": [],
            "mAP": [],
            "mAR": [],
            "mAF1": []
        },
        "val": {
            "loss": [],
            "mAP": [],
            "mAR": [],
            "mAF1": []
        }
    }

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_metrics = {i: [0, 0, 0, 0] for i in range(data.NUM_CLASSES)}
        total_samples = 0

        for phase in ["train", "val"]:
            with tqdm(dataloaders[phase], desc=f"{phase} epoch {epoch + 1}/{num_epochs}") as pbar:
                for x, y in pbar:
                    pred = model.forward(x)
                    loss, dloss = criterion(y, pred)
                    
                    running_loss += loss * y.shape[0]
                    
                    running_metrics = F.calculate_running_metrics(running_metrics, F.OHE_to_indeces(y), F.OHE_to_indeces(pred), data.NUM_CLASSES)
                    total_samples += y.shape[0]

                    if phase == "train":
                        model.backward(dloss)
                        
                        new_param = optimizer.step(model.parameters(), model.gradients())
                        
                        model.update_params(*new_param)

                    model.zero_grad()
                    
                    epoch_loss = running_loss / total_samples
                    epoch_metrics = []
                    for k, v in running_metrics.items():
                        epoch_metrics.append(v)
                    mAP, mAR, mAF1 = F.calculate_aggregated_metrics(epoch_metrics)
                    
                    pbar.set_postfix({
                        "loss": f"{epoch_loss:.4f}",
                        "mAP": f"{mAP:.4f}",
                        "mAR": f"{mAR:.4f}",
                        "mAF1": f"{mAF1:.4f}"
                    })
                    pbar.update(1)
            
            results[phase]["loss"].append(loss)
            results[phase]["mAP"].append(loss)
            results[phase]["mAR"].append(loss)
            results[phase]["mAF1"].append(loss)
            
            with open(os.path.join(save_path, "results.json"), "w") as file:
                json.dump(results, file)
        
        if save_period > 0 and (epoch + 1) % save_period == 0:
            model.save(os.path.join(save_path, "weights", f"model_{epoch + 1}.plk"))
    
    with open(os.path.join(save_path, "results.json"), "w") as file:
        json.dump(results, file)
    model.save(os.path.join(save_path, "weights", "model_last.plk"))


def main() -> None:
    batch_size = 64

    data_dir = "data"
    train_data, train_labels = unpickle_all(data_dir, ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"])
    test_data, test_labels = unpickle_all(data_dir, ["test_batch"])
    
    datasets = {"train": data.Dataset(train_data, train_labels),
                "val": data.Dataset(test_data, test_labels)}
    
    dataloaders = {"train": data.DataLoader(datasets["train"], batch_size=batch_size),
                   "val": data.DataLoader(datasets["val"], batch_size=batch_size)}
    
    model = M.CNNSmall()
    model.load("runs/12062025_1602/weights/model_5.plk")
    optimizer = optim.SGD(0.01)
    
    train(model, criterion.CEL, optimizer, dataloaders, 10, "runs", 5)


if __name__ == "__main__":
    main()
