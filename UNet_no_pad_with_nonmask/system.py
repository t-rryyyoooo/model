from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch import nn
from .model import UNetModel
from .dataset import UNetDataset
from .transform import UNetTransform
from torch.utils.data import DataLoader
from .utils import DICE
from .loss import WeightedCategoricalCrossEntropy
from .callbacks import EveryEpochModelCheckpoint, LatestModelCheckpoint, BestModelCheckpoint

class UNetSystem(pl.LightningModule):
    def __init__(self, dataset_mask_path, dataset_nonmask_path, log_path, criteria, rate, in_channel, num_class, learning_rate, batch_size, num_workers, dropout=0.5):
        super(UNetSystem, self).__init__()
        self.dataset_mask_path    = dataset_mask_path
        self.dataset_nonmask_path = dataset_nonmask_path
        self.num_class            = num_class
        self.model                = UNetModel(in_channel, self.num_class, dropout=dropout)
        self.criteria             = criteria
        self.rate                 = rate
        self.batch_size           = batch_size
        self.learning_rate        = learning_rate
        self.num_workers          = num_workers
        self.DICE                 = DICE(self.num_class)
        if self.num_class == 1:
            #self.loss             = nn.BCEWithLogitsLoss()
            self.loss             = nn.BCELoss()
        else:
            self.loss             = WeightedCategoricalCrossEntropy()
        self.callbacks            = [
                                    LatestModelCheckpoint(log_path),
                                    BestModelCheckpoint(log_path),
                                    EveryEpochModelCheckpoint(log_path)
                                    ]

    def forward(self, x):
        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        """
        label : not onehot 
        """
        image, label = batch
        image = image.float()
        label = label.float()

        pred = self.forward(image)

        """ Onehot for loss. """
        if self.num_class == 1:
            pred = torch.squeeze(pred)
            pred_argmax = (pred > 0.5).float()
            label = torch.squeeze(label)
            label_onehot = label
        else:
            pred_argmax = pred.argmax(dim=1)

            label_onehot = torch.eye(self.num_class)[label].permute((0, 4, 1, 2, 3))

        dice = self.DICE.compute(label, pred_argmax)
        loss = self.loss(pred, label_onehot)

        self.log("loss", loss, on_step=False, on_epoch=True)
        self.log("dice", dice, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        label : not onehot 
        """
        image, label = batch
        image = image.float()
        label = label.float()

        pred = self.forward(image)
        
        """ Onehot for loss. """
        if self.num_class == 1:
            pred = torch.squeeze(pred)
            pred_argmax = (pred > 0.5).float()
            label = torch.squeeze(label)
            label_onehot = label
        else:
            pred_argmax = pred.argmax(dim=1)

            label_onehot = torch.eye(self.num_class)[label].permute((0, 4, 1, 2, 3))


        dice = self.DICE.compute(label, pred_argmax)
        loss = self.loss(pred, label_onehot)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True)

        return loss

    def validation_epoch_end(self, outputs):
        avg = torch.stack([x for x in outputs]).mean()

        for callback in self.callbacks:
            callback(avg, self.model, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def train_dataloader(self):
        train_dataset = UNetDataset(
                dataset_mask_path = self.dataset_mask_path,
                dataset_nonmask_path = self.dataset_nonmask_path,
                phase = "train", 
                criteria = self.criteria,
                rate = self.rate,
                transform = UNetTransform()
                )

        train_loader = DataLoader(
                train_dataset,
                shuffle=True, 
                batch_size = self.batch_size, 
                num_workers = self.num_workers
                )

        return train_loader

    def val_dataloader(self):
        val_dataset = UNetDataset(
                dataset_mask_path = self.dataset_mask_path,
                dataset_nonmask_path = self.dataset_nonmask_path,
                phase = "val", 
                criteria = self.criteria,
                rate = self.rate,
                transform = UNetTransform()
                )

        val_loader = DataLoader(
                val_dataset, 
                batch_size = self.batch_size,
                num_workers = self.num_workers
                )

        return val_loader








