import cloudpickle
from pathlib import Path
import cv2
import numpy

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
    def __init__(self, save_directory: str, input_image_list: list, num_columns=5, save_ext="jpg"):
        """ Predict fed images and save them in num_columns * (len(input_image_list) // num_colum) arrangement.

        Parameters: 
            save_directory (str)    -- Save predicted images to this.
            input_image_list (list) -- Images fed to model.
            num_columns (int)       -- The number of images lined up horizontally.
            save_ext (str)          -- Extension for saving. npy or extensions available in cv2.
        """

        self.save_directory   = Path(save_directory)
        self.input_image_list = input_image_list

    def __call__(self, pred, model, epoch):
        pred_list = []
        temp_pred_list = []
        for i, input_image in enumerate(input_image_list):
            pred_array = model(input_image).to("cpu").detach().numpy().astype(np.float)
            pred_array = np.squeeze(pred_array)

            if (i + 1) % num_columns == 0:
                pred_list.append(temp_pred_list)
                temp_pred_list = []
                temp_pred_list.append(pred_array)

            else:
                temp_pred_list.append(pred_array)

        save_image = np.concatenate(pred_list)

        save_path = save_directory / "epoch_{:3d}_loss_{:.3f}.{}".format(epoch, pred, self.save_ext)

        if self.save_ext == "npy":
            np.save(str(save_path), save_image)

        else:
            cv2.imwrite(str(save_path), save_image)
