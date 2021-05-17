import cloudpickle
from pathlib import Path
import cv2
import numpy as np
import datetime
from torch.utils.data import DataLoader
from .transform import Pix2Pix3dTransform 
from .dataset import Pix2Pix3dDataset

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

        dataset = Pix2Pix3dDataset(
                    dataset_path = dataset_path,
                    criteria     = criteria,
                    transforms   = transforms,
                    phase        = phase
                    )

        self.data_loader = DataLoader(dataset)
        self.num_columns = num_columns
        self.save_ext    = save_ext

    def concatImages(self, data_loader, num_columns=5, model=None, input_or_target=0):
        pred_list = []
        temp_pred_list = []
        for i, arrays in enumerate(data_loader):
            arrays = [array.float() for array in arrays]
            if model is not None:
                pred_array = model(arrays[0]).to("cpu").detach().numpy().astype(np.float)
                pred_array = np.squeeze(pred_array)

            else:
                pred_array = np.squeeze(arrays[input_or_target])

            if pred_array.ndim > 2:
                d = pred_array.size()[0] - 1
                pred_array = pred_array[d//2 : (d+1)//2*-1, ...]
                pred_array = np.squeeze(pred_array)


            if (i + 1) % num_columns == 0:
                temp_pred_list.append(pred_array)
                temp_pred_list = np.concatenate(temp_pred_list, axis=1)

                pred_list.append(temp_pred_list)
                temp_pred_list = []

            else:
                temp_pred_list.append(pred_array)

        pred_image = np.concatenate(pred_list)

        return pred_image

    def saveImage(self, img, save_path):
        if self.save_ext == "npy":
            np.save(save_path, img)
        else:
            img = np.clip(img * 255, 0, 255)
            cv2.imwrite(save_path, img)



    def __call__(self, pred, model, epoch):
        dt_now = datetime.datetime.now()
        date = "{}_{}:{}".format(str(datetime.date.today()), dt_now.hour, dt_now.minute)

        save_pred = self.concatImages(self.data_loader, num_columns=self.num_columns, model=model)
        save_pred_path = self.save_directory / "{}_pred_epoch_{:03d}_loss_{:.3f}.{}".format(date, int(epoch), pred, self.save_ext)

        self.saveImage(save_pred, str(save_pred_path))
        
        if epoch == 0:
            save_img = self.concatImages(self.data_loader, num_columns=self.num_columns, model=None, input_or_target=0)
            save_tar = self.concatImages(self.data_loader, num_columns=self.num_columns, model=None, input_or_target=1)
            save_img_path  = self.save_directory / "{}_input.{}".format(date, self.save_ext)
            save_tar_path  = self.save_directory / "{}_target.{}".format(date, self.save_ext)

            self.saveImage(save_img, str(save_img_path))
            self.saveImage(save_tar, str(save_tar_path))

