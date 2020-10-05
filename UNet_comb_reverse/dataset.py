import torch.utils.data as data
if __name__ == "__main__":
    from utils import separateData
else:
    from .utils import separateData
from pathlib import Path

class UNetDataset(data.Dataset):
    def __init__(self, image_path_list, label_path, phase="train", criteria=None, transform=None):
        self.transform = transform
        self.phase = phase

        self.data_list = separateData(image_path_list, label_path, criteria, phase)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path = self.data_list[index]
        imageArray, labelArray = self.transform(self.phase, *path)

        return imageArray, labelArray

if __name__ == "__main__":
    from transform import UNetTransform
    transform = UNetTransform()

    dataset = UNetDataset(
            image_path_list = ["/sandisk/data/patch/Abdomen/no_pad/132-132-116-1/original/mask/image", "/sandisk/data/patch/Abdomen/with_pad/feature_map/48-48-32/original/mask/fold1"],
            label_path = "/sandisk/data/patch/Abdomen/no_pad/132-132-116-1/original/mask/image",
            phase="train",
            criteria={
                "train" : ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
                "val" : ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]
                },
            transform = transform
            )

    print(dataset.__len__())
    print(dataset.__getitem__(5))
    a, b = dataset.__getitem__(10)
    for xx in a:
        print(xx.shape)
    print(b.shape)




