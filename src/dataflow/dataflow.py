import numpy as np

import src.functional.functional as F


NUM_CLASSES = 10
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class Dataset:
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx: int | np.ndarray) -> tuple:
        return self.data[idx], self.labels[idx]
    

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, num_classes: int = NUM_CLASSES, shuffle=True):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.indices = np.array(list(range(self.dataset_size)), dtype=np.uint32)
        
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.batch_count = self.dataset_size // batch_size
    
    def __len__(self):
        return self.batch_count

    def __iter__(self):
        for i in range(self.batch_count):
            batch_indices = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
            X, y = self.dataset[batch_indices]
            y = F.BatchOHE(y, self.batch_size, self.num_classes)

            yield X, y
    
