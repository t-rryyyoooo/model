import cloudpickle
from pathlib import Path

class EveryEpochModelCheckpoint(object):
    def __init__(self, save_directory):
        self.save_directory= Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, pred, model, epoch):
        save_name = self.save_directory / "epoch_{:03d}_loss_{:.3f}.pkl".format(epoch, pred)
        with open(save_name, "wb") as f:
            cloudpickle.dump(model, f)


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
