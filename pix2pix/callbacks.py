import cloudpickle
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import DataLoader
from .transform import Pix2PixTransform 
from .dataset import Pix2PixDataset

class LatestModelCheckpoint(object):
    def __init__(self, save_directory, save_name="latest.pkl"):
        self.save_directory= Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.save_name = self.save_directory / save_name 


    def __call__(self, pred, model, epoch):
        with open(self.save_name, "wb") as f:
            
            cloudpickle.dump(model, f)

class BestModelCheckpoint(object):
    def __init__(self, save_directory, save_name="best.pkl"):
        self.best_value = 10**9
        self.save_directory= Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.save_name = self.save_directory / save_name 

    def __call__(self, pred, model, epoch):
        if pred < self.best_value:
            self.best_value = pred

            with open(self.save_name, "wb") as f:
                #print("Update best weight! loss : {}".format(self.best_value), flush=True)
                cloudpickle.dump(model, f)

class SavePredImages(object):
    def __init__(self, save_directory: str, dataset_path: str, criteria, transforms, phase="test", num_columns=5, save_ext="jpg"):
        """ Predict fed images and save them in num_columns * (len(input_image_list) // num_colum) arrangement.

        Parameters: 
            save_directory (str)    -- Save predicted images to this.
            dataset_path (str)      -- Dataset path which has data per patient in case_xx. 
            criteria (dict)         -- Which case_??? you use for train or val or test. {key : case_id("000")} (ex) {"train" : "000", "val" : "100", "test" : "test"}
            transforms              -- Preprocessing.
            phase (str)             -- The key for criteria.
            num_columns (int)       -- The number of images lined up horizontally.
            save_ext (str)          -- Extension for saving. npy or extensions available in cv2.
        """

        self.save_directory   = Path(save_directory)

        dataset = Pix2PixDataset(
                    dataset_path = dataset_path,
                    criteria     = criteria,
                    transforms   = transforms,
                    phase        = phase
                    )

        self.data_loader = DataLoader(
                            dataset
                            )
        self.num_columns = num_columns
        self.save_ext    = save_ext

    def __call__(self, pred, model, epoch):
        pred_list = []
        temp_pred_list = []
        input_list = []
        temp_input_list = []

        for i, (input_image_array, _) in enumerate(self.data_loader):
            input_image_array = input_image_array.float()
            input_array = input_image_array.to("cpu").detach().numpy().astype(np.float)
            input_array = np.squeeze(input_array)
            pred_array  = model(input_image_array).to("cpu").detach().numpy().astype(np.float)
            pred_array  = np.squeeze(pred_array)

            if (i + 1) % self.num_columns == 0:
                temp_pred_list.append(pred_array)
                temp_pred_list = np.concatenate(temp_pred_list)
                pred_list.append(temp_pred_list)
                temp_pred_list = []

                temp_input_list.append(input_array)
                temp_input_list = np.concatenate(temp_input_list)
                input_list.append(temp_input_list)
                temp_input_list = []

            else:
                temp_pred_list.append(pred_array)
                temp_input_list.append(input_array)

        save_pred = np.concatenate(pred_list, axis=1)
        save_img  = np.concatenate(input_list, axis=1)

        save_pred_path = self.save_directory / "pred_epoch_{:03d}_loss_{:.3f}.{}".format(int(epoch), pred, self.save_ext)
        save_img_path  = self.save_directory / "input_epoch_{:03d}_loss_{:.3f}.{}".format(int(epoch), pred, self.save_ext)

        if self.save_ext == "npy":
            np.save(str(save_pred_path), save_pred)
            np.save(str(save_img_path), save_img)

        else:
            save_pred = np.clip(save_pred * 255, 0, 255)
            save_img  = np.clip(save_img * 255, 0, 255)
            cv2.imwrite(str(save_pred_path), save_pred)
            cv2.imwrite(str(save_img_path), save_img)
