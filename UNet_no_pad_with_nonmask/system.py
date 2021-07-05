from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch import nn
from .model import UNetModel
from .dataset import UNetDataset
from .transform import UNetTransform
from torch.utils.data import DataLoader
from .utils import DICEPerClass
from .loss import WeightedCategoricalCrossEntropy, DiceBCELoss
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
        self.DICE                 = DICEPerClass()
        if self.num_class == 1:
            #self.loss             = nn.BCEWithLogitsLoss()
            #self.loss             = nn.BCELoss()
            self.loss             = DiceBCELoss()
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
        loss, dice = self.calcLossAndDICE(batch)

        self.logging(loss, dice)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice = self.calcLossAndDICE(batch)

        self.logging(loss, dice, for_val=True)

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
                transform = UNetTransform(
                                num_class = self.num_class
                                )
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
                transform = UNetTransform(
                                num_class = self.num_class
                                )
                )

        val_loader = DataLoader(
                val_dataset, 
                batch_size = self.batch_size,
                num_workers = self.num_workers
                )

        return val_loader

    def calcDICE(self, pred, label):
        if self.num_class == 1:
            pred_onehot = (pred > 0.5)
        else:
            pred_onehot = torch.eye(self.num_class)[pred.argmax(dim=1)].permute((0, 4, 1, 2, 3)).to(self.device)

        dice = self.DICE(pred_onehot, label)

        return dice

    def calcLossAndDICE(self, batch):
        image, label = batch

        image = image.float()
        label = label.float()

        pred = self.forward(image)

        loss = self.loss(pred, label)
        dice = self.calcDICE(pred, label)

        return loss, dice

    def logging(self, loss, dice, for_val=False):
        for i in range(len(dice)):
            if for_val:
                dice_tag = "val_dice_{}".format(i)
            else:
                dice_tag = "dice_{}".format(i)

            self.log(dice_tag, dice[i], on_step=False, on_epoch=True)

        if for_val:
            loss_tag = "val_loss"
        else:
            loss_tag = "loss"

        self.log(loss_tag, loss, on_step=False, on_epoch=True)
