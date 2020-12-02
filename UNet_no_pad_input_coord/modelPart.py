import torch
from torch import nn
from .utils import cropping3D

class FirstConvBlock(nn.Module):
    def __init__(self, in_channel_img, in_channel_coord, mid_channel, out_channel, n=2, use_bn=True, apply_pooling=True):
        super(FirstConvBlock, self).__init__()
        self.apply_pooling = apply_pooling
        self.use_bn = use_bn

        self.conv_img = nn.Conv3d(in_channel_img, mid_channel, (3, 3, 3))
        self.conv_coord = nn.Conv3d(in_channel_coord, mid_channel, (1, 1, 1))

        if self.use_bn:
            self.bn_1 = nn.BatchNorm3d(mid_channel)

        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv3d(mid_channel, out_channel, (3, 3, 3))

        if self.use_bn:
            self.bn_2 = nn.BatchNorm3d(out_channel)

        self.relu_2 = nn.ReLU()

        if self.apply_pooling:
            self.maxpool = nn.MaxPool3d((2, 2, 2))
        
    def forward(self, x_img, x_coord):
        # x_img shape : b * mid_ch * z * y * x
        x_img = self.conv_img(x_img) 
        # x_coord shape: b * ch * z * y * x
        x_coord = self.conv_coord(x_coord) 
        c = [i - j for (i, j) in zip(x_coord.size()[2:], x_img.size()[2:])]
        x_coord = cropping3D(x_coord, (c[0]//2, (c[0] + 1)//2), (c[1]//2, (c[1] + 1)//2), (c[2]//2, (c[2] + 1)//2))
 
        x = x_img + x_coord
        if self.use_bn:
            x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        if self.use_bn:
            x = self.bn_2(x)
        x = self.relu_2(x)

        conv_result = x

        if self.apply_pooling:
            x = self.maxpool(x)

        return x, conv_result


class DoubleConvolution(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, n=2, use_bn=True):
        super(DoubleConvolution, self).__init__()

        self.layers = []
        for i in range(1, n + 1):
            if i == 1:
                x = nn.Conv3d(in_channel, mid_channel, (3, 3, 3))
            else:
                x = nn.Conv3d(mid_channel, out_channel, (3, 3, 3))

            self.layers.append(x)

            if use_bn:
                if i == 1:
                    self.layers.append(nn.BatchNorm3d(mid_channel))
                else:
                    self.layers.append(nn.BatchNorm3d(out_channel))
                
            
            self.layers.append(nn.ReLU())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class CreateConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, n=2, use_bn=True, apply_pooling=True):
        super(CreateConvBlock, self).__init__()
        self.apply_pooling = apply_pooling

        self.DoubleConvolution = DoubleConvolution(in_channel, mid_channel, out_channel, n=2, use_bn=use_bn)

        if apply_pooling:
            self.maxpool = nn.MaxPool3d((2, 2, 2))
        
    def forward(self, x):
        x = self.DoubleConvolution(x)
        conv_result = x
        if self.apply_pooling:
            x = self.maxpool(x)

        return x, conv_result


class CreateUpConvBlock(nn.Module):
    def __init__(self, in_channel, concat_channel, mid_channel, out_channel,  n=2, use_bn=True):
        super(CreateUpConvBlock, self).__init__()

        x = nn.ConvTranspose3d(in_channel, in_channel, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), dilation=1)
        self.convTranspose = x

        self.DoubleConvolution = DoubleConvolution(in_channel + concat_channel, mid_channel, out_channel, n=2, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.convTranspose(x1)
        c = [(i - j) for (i, j) in zip(x2.size()[2:], x1.size()[2:])]
        hc = [ i // 2 for i in c]

        x2 = cropping3D(x2, (hc[0], hc[0] if c[0]%2 == 0 else hc[0] + 1), (hc[1], hc[1] if c[1]%2 == 0 else hc[1] + 1), (hc[2], hc[2] if c[2]%2 == 0 else hc[2] + 1))
        x = torch.cat([x2, x1], dim=1)

        x = self.DoubleConvolution(x)

        return x


