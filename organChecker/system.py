import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import OrganCheckerDataset
from .transform import OrganCheckerTransform
from .callbacks import LatestModelCheckpoint, BestModelCheckpoint
from .utils import recall, defineModel


class OrganCheckerSystem(pl.LightningModule):
    """ Define ResNet50 learning flow. """

    def __init__(self, dataset_path=None, criteria=None, log_path=None, in_ch=1, model_name="resnet50", lr=0.001, batch_size=3, num_workers=6, gpu_ids=[]):
        super(OrganCheckerSystem, self).__init__()

        self.dataset_path  = dataset_path
        self.criteria      = criteria
        self.callbacks     = [
                            LatestModelCheckpoint(log_path),
                            BestModelCheckpoint(log_path)
                            ]
        self.model         = defineModel(model_name, in_ch=in_ch)
        self.batch_size    = batch_size
        self.lr            = lr
        self.num_workers   = num_workers
        self.loss          = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, label = batch
        image = image.float()
        label = label.float()

        pred = torch.squeeze(self.forward(image))

        pred_onehot = (pred > 0.5).float()
        
        loss         = self.loss(pred, label)
        recall_score = recall(pred_onehot, label)

        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", recall_score, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        image = image.float()
        label = label.float()

        pred = torch.squeeze(self.forward(image))

        pred_onehot = (pred > 0.5).float()
        
        loss   = self.loss(pred, label)
        recall_score = recall(pred_onehot, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall", recall_score, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        for callback in self.callbacks:
            callback(avg_loss, self.model, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        return optimizer

    def train_dataloader(self):
        train_dataset = OrganCheckerDataset(
                            dataset_path = self.dataset_path,
                            phase = "train",
                            criteria = self.criteria,
                            transforms = OrganCheckerTransform()
                            )

        train_loader = DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size = self.batch_size,
                        num_workers = self.num_workers
                        )

        return train_loader

    def val_dataloader(self):
        val_dataset = OrganCheckerDataset(
                            dataset_path = self.dataset_path,
                            phase = "val",
                            criteria = self.criteria,
                            transforms = OrganCheckerTransform()
                            )

        val_loader = DataLoader(
                        val_dataset,
                        batch_size = self.batch_size,
                        num_workers = self.num_workers
                        )

        return val_loader
