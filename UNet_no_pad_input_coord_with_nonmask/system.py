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

class UNetSystem(pl.LightningModule):
    def __init__(self, dataset_mask_path, dataset_nonmask_path, criteria, rate, in_channel_img, in_channel_coord, num_class, learning_rate, batch_size, checkpoint, num_workers, dropout=0.5):
        super(UNetSystem, self).__init__()
        use_cuda = torch.cuda.is_available() and True
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dataset_mask_path = dataset_mask_path
        self.dataset_nonmask_path = dataset_nonmask_path
        self.num_class = num_class
        self.model = UNetModel(in_channel_img, in_channel_coord, self.num_class, dropout=dropout).to(self.device, dtype=torch.float)
        self.criteria = criteria
        self.rate = rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint = checkpoint
        self.num_workers = num_workers
        self.DICE = DICE(self.num_class, self.device)
        #self.loss = WeightedCategoricalCrossEntropy(device=self.device)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x_img, x_coord):
        x = self.model(x_img, x_coord)

        return x

    def training_step(self, batch, batch_idx):
        """
        label : not onehot 
        """
        images, label = batch
        images = [image.to(self.device, dtype=torch.float) for image in images]
        label = label.to(self.device, dtype=torch.long)

        pred = self.forward(*images).to(self.device)
        

        """ Onehot for loss. """
        pred_argmax = pred.argmax(dim=1)
        label_onehot = torch.eye(self.num_class)[label].to(self.device).permute((0, 4, 1, 2, 3))

        dice = self.DICE.compute(label, pred_argmax)
        loss = self.loss(pred, label_onehot)
        
        return {"loss" : loss, "dice" : dice}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_dice = torch.stack([x["dice"] for x in outputs]).mean()

        self.checkpoint(avg_loss.item(), self.model)

        tensorboard_logs = {
                "loss" : avg_loss,
                "dice" : avg_dice, 
                }
        progress_bar = {
                "loss" : avg_loss,
                "dice" : avg_dice
                }


        return {"avg_loss" : avg_loss, "log" : tensorboard_logs, "progress_bar" : progress_bar}


    def validation_step(self, batch, batch_idx):
        """
        label : not onehot 
        """
        images, label = batch
        images = [image.to(self.device, dtype=torch.float) for image in images]
        label = label.to(self.device, dtype=torch.long)

        pred = self.forward(*images).to(self.device)
        

        """ Onehot for loss. """
        pred_argmax = pred.argmax(dim=1)
        label_onehot = torch.eye(self.num_class)[label].to(self.device).permute((0, 4, 1, 2, 3))

        dice = self.DICE.compute(label, pred_argmax)
        loss = self.loss(pred, label_onehot)

        return {"val_loss" : loss, "val_dice" : dice}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_dice = torch.stack([x["val_dice"] for x in outputs]).mean()

        self.checkpoint(avg_loss.item(), self.model)

        tensorboard_logs = {
                "val_loss" : avg_loss,
                "val_dice" : avg_dice, 
                }
        progress_bar = {
                "val_loss" : avg_loss,
                "val_dice" : avg_dice
                }


        return {"avg_val_loss" : avg_loss, "log" : tensorboard_logs, "progress_bar" : progress_bar}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    @pl.data_loader
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

    @pl.data_loader
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








