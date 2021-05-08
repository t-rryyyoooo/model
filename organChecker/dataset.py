import torch.utils.data as data
from .utils import separateData

class OrganCheckerDataset(data.Dataset):
    def __init__(self, dataset_path, criteria, transforms, phase="train"):
        self.transforms  = transforms
        self.phase       = phase

        self.data_list = separateData(dataset_path, criteria, phase)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path = self.data_list[index]
        input_image_array, target_image_array = self.transforms(self.phase, *path)

        return input_image_array, target_image_array
