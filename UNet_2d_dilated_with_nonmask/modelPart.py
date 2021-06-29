import torch
from torch import nn

class CreateConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_bn=True, apply_pooling=True, pooling_size=2):
        super(CreateConvBlock, self).__init__()
        self.apply_pooling = apply_pooling
        self.use_bn        = use_bn

        self.conv_layer = nn.Conv2d(in_channel, out_channel, 3, padding=2, dilation=2)

        if use_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()

        if apply_pooling:
            self.maxpool = nn.MaxPool2d(pooling_size)
        
    def forward(self, x):
        x = self.conv_layer(x)
        if self.use_bn:
            x = self.bn(x)
        
        x = self.relu(x)

        conv_result = x
        if self.apply_pooling:
            x = self.maxpool(x)

        return x, conv_result


class CreateUpConvBlock(nn.Module):
    def __init__(self, in_channel, concat_channel, out_channel,  n=2, use_bn=True, upsampling_size=2):
        super(CreateUpConvBlock, self).__init__()
        self.use_bn = use_bn

        self.conv_transpose = nn.ConvTranspose2d(in_channel, in_channel, upsampling_size, stride=upsampling_size, padding=0, dilation=1)

        self.conv_layer = nn.Conv2d(in_channel + concat_channel, out_channel, 3, padding=2, dilation=2)

        if use_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.conv_transpose(x1)
        c = [(i - j) for (i, j) in zip(x2.size()[2:], x1.size()[2:])]

        x1 = nn.functional.pad(x1, (c[0] // 2, (c[0] * 2 + 1) // 2, c[1] // 2, (c[1] * 2 + 1) // 2))
        

        x = torch.cat([x2, x1], dim=1)

        x = self.conv_layer(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)

        return x
