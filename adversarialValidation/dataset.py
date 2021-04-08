from pathlib import Path
from torch.utils import data
import numpy as np

class CNNDataset(data.Dataset):
    def __init__(self, dataset, phase="train", transform=None):
        self.transform = transform
        self.phase = phase
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path, label = self.dataset[index]
        imageArray = self.transform(self.phase, path)
        label = label[np.newaxis, ...]

        return imageArray, label


