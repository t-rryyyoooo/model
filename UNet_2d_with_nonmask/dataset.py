import torch.utils.data as data
from .utils import separateDataWithNonMask
from pathlib import Path

class UNetDataset(data.Dataset):
    def __init__(self, dataset_mask_path, dataset_nonmask_path, criteria, rate, transform, phase="train"):
        self.transform = transform
        self.phase = phase

        self.data_list = separateDataWithNonMask(dataset_mask_path, dataset_nonmask_path, criteria, phase, rate)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path = self.data_list[index]
        imageArray, labelArray = self.transform(self.phase, *path)

        return imageArray, labelArray


