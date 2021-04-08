from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch import nn
from .model import CNN
from .dataset import CNNDataset
from .transform import CNNTransform
from torch.utils.data import DataLoader
from .loss import WeightedCategoricalCrossEntropy
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score

class CNNSystem(pl.LightningModule):
    def __init__(self, dataset, in_channel, out_channel, learning_rate, batch_size, checkpoint, num_workers, dropout=0.2):
        super(CNNSystem, self).__init__()
        use_cuda = torch.cuda.is_available() and True
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.dataset = dataset
        self.model = CNN(in_channel, out_channel, dropout=dropout).to(self.device, dtype=torch.float)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint = checkpoint
        self.num_workers = num_workers
        #self.loss = WeightedCategoricalCrossEntropy(device=self.device)
        self.loss = BCEWithLogitsLoss()

    def forward(self, x):
        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        image, label = batch
        image = image.to(self.device, dtype=torch.float)
        label = label.to(self.device, dtype=torch.float)

        pred = self.forward(image).to(self.device)

        loss = self.loss(pred, label)
        
        pred_argmax = (pred > 0.5).to(self.device, dtype=torch.int)
        acc = torch.sum(label == pred_argmax) * 1.0 / len(label)

        return {"loss" : loss, "acc" : acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()

        self.checkpoint(avg_loss.item(), self.model)

        tensorboard_logs = {
                "loss" : avg_loss,
                "acc" : avg_acc,
                }
        progress_bar = {
                "loss" : avg_loss,
                "acc" : avg_acc,
                }


        return {"avg_loss" : avg_loss, "log" : tensorboard_logs, "progress_bar" : progress_bar}


    def validation_step(self, batch, batch_idx):
        image, label = batch
        image = image.to(self.device, dtype=torch.float)
        label = label.to(self.device, dtype=torch.float)

        pred = self.forward(image).to(self.device)

        loss = self.loss(pred, label)

        pred_argmax = (pred > 0.5).to(self.device, dtype=torch.int)
        acc = torch.sum(label == pred_argmax) * 1.0 / len(label)

        return {"val_loss" : loss, "val_acc" : acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        self.checkpoint(avg_loss.item(), self.model)

        tensorboard_logs = {
                "val_loss" : avg_loss,
                "val_acc" : avg_acc,
                }
        progress_bar = {
                "val_loss" : avg_loss,
                "val_acc" : avg_acc,
                }


        return {"avg_val_loss" : avg_loss, "log" : tensorboard_logs, "progress_bar" : progress_bar}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        train_dataset = CNNDataset(
                dataset = self.dataset, 
                phase = "train", 
                transform = CNNTransform()
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
        val_dataset = CNNDataset(
                dataset= self.dataset, 
                phase = "val", 
                transform = CNNTransform()
                )

        val_loader = DataLoader(
                val_dataset, 
                batch_size = self.batch_size,
                num_workers = self.num_workers
                )

        return val_loader








