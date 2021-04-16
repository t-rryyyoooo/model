from pathlib import Path
import torch.utils.data as data
if __name__ == "__main__":
    from utils import separateData
else:
    from .utils import separateData

class Pix2PixDataset(data.Dataset):
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


#Test
if __name__ == "__main__":
    import numpy as np
    from transform import Pix2PixTransform
    a = np.array(1)
    save_path = Path("/Users/tanimotoryou/Desktop/test")
    transforms = Pix2PixTransform(4, 4)
    criteria = {
            "train" : ["00", "01"],
            "val"   : ["02", "03"]
            }

    for pid in range(4):
        save = save_path / "case_0{}".format(str(pid))
        save.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            s_i = save / "input_000{}.npy".format(str(i))
            s_t = save / "target_000{}.npy".format(str(i))
            np.save(str(s_i), a)
            np.save(str(s_t), a)

    d_t = Pix2PixDataset(str(save_path), criteria, transforms, phase="train")
    d_v = Pix2PixDataset(str(save_path), criteria, transforms, phase="val")
    print(d_t.__len__(), d_v.__len__())
    print(d_t.__getitem__(3)[0].shape, d_v.__getitem__(5)[1].shape)
