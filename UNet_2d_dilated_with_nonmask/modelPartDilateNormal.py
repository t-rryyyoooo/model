import torch
from torch import nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, n=2, use_bn=True):
        super(DoubleConvolution, self).__init__()

        self.layers = []
        for i in range(1, n + 1):
            if i == 1:
                x = nn.Conv2d(in_channel, mid_channel, 3, padding=2, dilation=2)
            else:
                x = nn.Conv2d(mid_channel, out_channel, 3, padding=1, dilation=1)

            self.layers.append(x)

            if use_bn:
                if i == 1:
                    self.layers.append(nn.BatchNorm2d(mid_channel))
                else:
                    self.layers.append(nn.BatchNorm2d(out_channel))
                
            
            self.layers.append(nn.ReLU())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class CreateConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, n=2, use_bn=True, apply_pooling=True, pooling_size=2):
        super(CreateConvBlock, self).__init__()
        self.apply_pooling = apply_pooling

        self.DoubleConvolution = DoubleConvolution(in_channel, mid_channel, out_channel, n=2, use_bn=use_bn)

        if apply_pooling:
            self.maxpool = nn.MaxPool2d(pooling_size)
        
    def forward(self, x):
        x = self.DoubleConvolution(x)
        conv_result = x
        if self.apply_pooling:
            x = self.maxpool(x)

        return x, conv_result


class CreateUpConvBlock(nn.Module):
    def __init__(self, in_channel, concat_channel, mid_channel, out_channel,  n=2, use_bn=True, upsampling_size=2):
        super(CreateUpConvBlock, self).__init__()

        x = nn.ConvTranspose2d(in_channel, in_channel, upsampling_size, stride=upsampling_size, padding=0, dilation=1)
        self.convTranspose = x

        self.DoubleConvolution = DoubleConvolution(in_channel + concat_channel, mid_channel, out_channel, n=2, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.convTranspose(x1)
        c = [(i - j) for (i, j) in zip(x2.size()[2:], x1.size()[2:])]

        x1 = nn.functional.pad(x1, (c[0] // 2, (c[0] * 2 + 1) // 2, c[1] // 2, (c[1] * 2 + 1) // 2))
        

        x = torch.cat([x2, x1], dim=1)

        x = self.DoubleConvolution(x)

        return x
