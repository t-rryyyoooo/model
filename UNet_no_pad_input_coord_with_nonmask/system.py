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
from .callbacks import EveryEpochModelCheckpoint, LatestModelCheckpoint, BestModelCheckpoint
from .loss import WeightedCategoricalCrossEntropy, DICEPerClassLoss

class UNetSystem(pl.LightningModule):
    def __init__(self, dataset_mask_path, dataset_nonmask_path, log_path, criteria, rate, in_channel_img, in_channel_coord, num_class, learning_rate, batch_size, num_workers, dropout=0.5, ambience=False):
        super(UNetSystem, self).__init__()
        use_cuda                  = torch.cuda.is_available() and True
        self.dataset_mask_path    = dataset_mask_path
        self.dataset_nonmask_path = dataset_nonmask_path
        self.num_class            = num_class
        self.model                = UNetModel(in_channel_img, in_channel_coord, self.num_class, dropout=dropout)
        self.criteria             = criteria
        self.rate                 = rate
        self.batch_size           = batch_size
        self.learning_rate        = learning_rate
        self.ambience             = ambience
        self.callbacks            = [
                EveryEpochModelCheckpoint(log_path),
                LatestModelCheckpoint(log_path),
                BestModelCheckpoint(log_path)
                ]
        self.num_workers          = num_workers
        self.DICE                 = DICEPerClass()
        self.loss                 = DICEPerClassLoss()
        #self.loss = WeightedCategoricalCrossEntropy()
        #self.loss = nn.CrossEntropyLoss()

    def forward(self, x_img, x_coord):
        x = self.model(x_img, x_coord)

        return x

    def training_step(self, batch, batch_idx):
        loss, dice = self.calcLossAndDICE(batch)

        self.logLossAndDICE(loss, dice)

        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, dice = self.calcLossAndDICE(batch)

        self.logLossAndDICE(loss, dice)

        return loss

    def validation_epoch_end(self, outputs):
        avg = torch.stack([x for x in outputs]).mean()

        for callback in self.callbacks:
            callback(avg.item(), self.model, self.current_epoch)


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

    def calculateDICE(self, pred, label):
        """
        pred  : Probability
        label : Onehot
        """
        pred_onehot = torch.eye(self.num_class)[pred.argmax(dim=1)].permute((0, 4, 1, 2, 3)).to(self.device)

        if self.ambience:
            label = torch.eye(self.num_class)[label.argmax(dim=1)].permute(0, 4, 1, 2, 3)

        dice = self.DICE(pred_onehot, label)

        return dice

    def calcLossAndDICE(self, batch):
        images, label = batch

        images = [image.float() for image in images]
        label  = label.float()

        pred = self.forward(*images)
        
        loss = self.loss(pred, label)
        dice = self.calculateDICE(pred, label)

        return loss, dice

    def logLossAndDICE(self, loss, dice):
        for i in range(len(dice)):
            self.log("dice_{}".format(i), dice[i], on_step=False, on_epoch=True)

        self.log("loss", loss, on_step=False, on_epoch=True)



