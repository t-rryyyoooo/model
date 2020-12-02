import torch
from torch import nn
if __name__ == "__main__":
    from utils import cropping3D
else:
    from .utils import cropping3D

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

class CreateFinalBlock(nn.Module):
    def __init__(self, in_channel, concat_channel_1, concat_channel_2, mid_channel, out_channel,  n=2, use_bn=True):
        super(CreateFinalBlock, self).__init__()

        x = nn.ConvTranspose3d(in_channel, in_channel, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), dilation=1)
        self.convTranspose = x

        self.DoubleConvolution = DoubleConvolution(in_channel + concat_channel_1 + concat_channel_2, mid_channel, out_channel, n=2, use_bn=use_bn)

    def forward(self, x1, x2, x3):
        x1 = self.convTranspose(x1)
        c = [(i - j) for (i, j) in zip(x2.size()[2:], x1.size()[2:])]
        hc = [ i // 2 for i in c]

        x2 = cropping3D(x2, (hc[0], hc[0] if c[0]%2 == 0 else hc[0] + 1), (hc[1], hc[1] if c[1]%2 == 0 else hc[1] + 1), (hc[2], hc[2] if c[2]%2 == 0 else hc[2] + 1))

        c = [(i - j) for (i, j) in zip(x3.size()[2:], x1.size()[2:])]
        hc = [ i // 2 for i in c]

        x3 = cropping3D(x3, (hc[0], hc[0] if c[0]%2 == 0 else hc[0] + 1), (hc[1], hc[1] if c[1]%2 == 0 else hc[1] + 1), (hc[2], hc[2] if c[2]%2 == 0 else hc[2] + 1))
        x = torch.cat([x3, x2, x1], dim=1)

        x = self.DoubleConvolution(x)

        return x
