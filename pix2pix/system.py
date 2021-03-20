from math import log10
import pytoch_lightning as pl
from torch import nn
import torch
from torch.utils.data import DataLoader
from .utils import defineG, defineD
from .dataset import Pix2PixDataset
from .transform import Pix2PixTransform


class Pix2PixSystem(pl.LightningModule):
    """ Define pix2pix learning flow. 
    Parameters: 

    """
    def __init__(self, dataset_path=None, criteria=None, checkpoint=None, lr=0.001, batch_size=3, num_workers=6, G_input_ch=1, G_output_ch=1, G_name="unet_256", D_input_ch=2, D_name="PatchGAN", ngf=64, gpu_ids=[]):
        super(Pix2PixSystem, self).__init__()

        self.dataset_path  = dataset_path
        self.criteria      = criteria
        self.batch_size    = batch_size
        self.lr            = lr
        self.num_workers   = num_workers
        self.generator     = defineG(
                                input_ch = G_input_ch, 
                                output_ch = G_output_ch, 
                                ngf = ngf, 
                                G_name = G_name, 
                                use_dropout = True, 
                                gpu_ids = gpu_ids
                                )
        self.discriminator = defineD(
                                input_ch = D_input_ch,
                                ndf = ngf,
                                D_name = D_name,
                                gpu_ids = gpu_ids
                                )
        self.loss_fun_gan  = GANLoss()
        self.loss_func_l1  = nn.L1Loss()
        self.loss_func_mse = nn.MSELoss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_input, real_target = batch
        fake = self.generator(real_input)
        real_fake = torch.cat((real_input, fake), 1)
        pred_fake = self.discriminator.forward(real_fake)

        # G
        if optimizer_idx == 0:
            g_loss_gan = self.loss_fun_gan(pred_fake, True)
            g_loss_l1  = self.loss_func_l1(fake, real_target)
            g_loss = g_loss_gan + g_loss_l1

            return g_loss

        # D
        if optimizer_idx == 1:
            d_loss_fake = self.loss_fun_gan(pred, False)

            real_real = torch.cat((real_input, real_target), 1)
            pred_real = self.discriminator(real_real)
            d_loss_real = self.loss_fun_gan(pred_real, True)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            return d_loss

    def validation_step(self, batch, batch_idx):
        real_input, real_target = batch
        fake = self.generator(real_input)
        mse = self.loss_func_mse(fake, real_target)
        pnsr = torch.tensor([10 * log10(1 / mse.item())])# Note: input image need to normalize (0 - 1). If not, change 1 to the maximize value in the image.
        return psnr

    def validation_epoch_end(self, outputs):
        avg_psnr = torch.stack(outputs).mean()

        """ Save model. """
        if self.checkpoint is not None:
            self.checkpoint(avg_psnr.item(), self)
        

        logs = {"val_psnr" : avg_psnr}

        return {"avg_val_loss" : avg_psnr, "log" : logs, "progress_bar" : logs}

    def configure_optimizeris(self):
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        self.d_optimizer = torch.optim.Adam(seld.discriminator.parameters(), lr=self.lr)
        
        optimizer_list = [self.g_optimizer, self.d_optimizer]
        return optimizer_list

    @pl.data_loader
    def train_dataloader(self):
        train_dataset = Pix2PixDataset(
                            dataset_path = self.dataset_path,
                            phase = "train",
                            criteria = self.criteria,
                            transform = Pix2PixTransform()
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
        val_dataset = Pix2PixDataset(
                            dataset_path = self.dataset_path,
                            phase = "val",
                            criteria = self.criteria,
                            )

        val_dataloader = DataLoader(
                        val_dataset,
                        shuffle=True,
                        batch_size = self.batch_size,
                        num_workers = self.num_workers
                        )

        return val_loader





    
