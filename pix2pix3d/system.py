from math import log10
import pytorch_lightning as pl
from torch import nn
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from .utils import defineG, defineD
from .dataset import Pix2Pix3dDataset
from .transform import Pix2Pix3dTransform
from .loss import GANLoss
from .callbacks import LatestModelCheckpoint, BestModelCheckpoint, SavePredImages


class Pix2Pix3dSystem(pl.LightningModule):
    """ Define pix2pix learning flow. 
    Parameters: 

    """
    def __init__(self, dataset_path=None, criteria=None, log_path=None, l1_lambda=100., lr=0.001, batch_size=3, num_workers=6, G_input_ch=1, G_output_ch=1, G_name="unet_64", D_input_ch=2, D_name="PatchGAN", D_n_layers=3,  ngf=64, gpu_ids=[]):
        super(Pix2Pix3dSystem, self).__init__()

        self.dataset_path  = dataset_path
        self.criteria      = criteria
        self.callbacks     = [
                            LatestModelCheckpoint(log_path),
                            BestModelCheckpoint(log_path),
                            #SavePredImages(log_path, dataset_path, criteria, Pix2Pix3dTransform(), phase="val")
                                ]
        self.batch_size    = batch_size
        self.l1_lambda     = l1_lambda
        self.lr            = lr
        self.num_workers   = num_workers
        self.generator     = defineG(
                                input_ch    = G_input_ch, 
                                output_ch   = G_output_ch, 
                                ngf         = ngf, 
                                G_name      = G_name, 
                                use_dropout = True, 
                                gpu_ids     = gpu_ids
                                )
        self.discriminator = defineD(
                                input_ch = D_input_ch,
                                ndf      = ngf,
                                D_name   = D_name,
                                n_layers = D_n_layers,
                                gpu_ids  = gpu_ids
                                )
        self.D_input_ch    = D_input_ch
        self.loss_fun_gan  = GANLoss()
        self.loss_func_l1  = nn.L1Loss()
        self.loss_func_mse = nn.MSELoss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_input, real_target = batch
        real_input  = real_input.float()
        real_target = real_target.float()
        fake = self.generator(real_input)
        
        real_fake = torch.cat((real_input, fake), 1)
        pred_fake = self.discriminator.forward(real_fake)

        # G
        if optimizer_idx == 0:
            g_loss_gan = self.loss_fun_gan(pred_fake, True)
            g_loss_l1  = self.loss_func_l1(fake, real_target) * self.l1_lambda
            g_loss = g_loss_gan + g_loss_l1

            self.log("g_loss", g_loss, on_step=False, on_epoch=True)

            return g_loss

        # D
        if optimizer_idx == 1:
            d_loss_fake = self.loss_fun_gan(pred_fake, False)

            real_real = torch.cat((real_input, real_target), 1)
            pred_real = self.discriminator(real_real)
            d_loss_real = self.loss_fun_gan(pred_real, True)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            self.log("d_loss", d_loss, on_step=False, on_epoch=True)

            return d_loss

    def validation_step(self, batch, batch_idx):
        real_input, real_target = batch
        real_input  = real_input.float()
        real_target = real_target.float()
        fake = self.generator(real_input)
        mse = self.loss_func_mse(fake, real_target)
        psnr = torch.tensor([10 * log10(1 / mse.item())])# Note: input image need to normalize (0 - 1). If not, change 1 to the maximize value in the image.
        self.log("PSNR", psnr, on_step=False)

        return psnr

    def validation_epoch_end(self, outputs):
        avg_psnr = torch.stack([x for x in outputs]).mean()
        for callback in self.callbacks:
            callback(avg_psnr, self.generator, self.current_epoch)

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        g_scheduler = StepLR(g_optimizer, step_size=50, gamma=0.5)
        d_scheduler = StepLR(d_optimizer, step_size=50, gamma=0.5)
        
        optimizer_list = [g_optimizer, d_optimizer]
        scheduler_list = [g_scheduler, d_scheduler]
        return optimizer_list#, scheduler_list

    def train_dataloader(self):
        train_dataset = Pix2Pix3dDataset(
                            dataset_path = self.dataset_path,
                            phase = "train",
                            criteria = self.criteria,
                            transforms = Pix2Pix3dTransform()
                            )

        train_loader = DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size = self.batch_size,
                        num_workers = self.num_workers
                        )

        return train_loader

    def val_dataloader(self):
        val_dataset = Pix2Pix3dDataset(
                            dataset_path = self.dataset_path,
                            phase = "val",
                            criteria = self.criteria,
                            transforms = Pix2Pix3dTransform()
                            )

        val_loader = DataLoader(
                        val_dataset,
                        shuffle=True,
                        batch_size = self.batch_size,
                        num_workers = self.num_workers
                        )

        return val_loader
