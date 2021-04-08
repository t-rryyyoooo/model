import torch
from torch import nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_bn=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, (3, 3, 3))

        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm3d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)

        if self.use_bn:
            x = self.bn(x)

        x = self.relu(x)

        return x

class DoubleConvolution(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, use_bn=True):
        super(DoubleConvolution, self).__init__()

        self.conv1 = ConvBlock(in_channel, mid_channel, use_bn)
        self.conv2 = ConvBlock(mid_channel, out_channel, use_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, use_maxpool=True, use_bn=True):
        super(CNNBlock, self).__init__()

        self.double_conv = DoubleConvolution(in_channel, mid_channel, out_channel)
        self.use_maxpool = use_maxpool
        if self.use_maxpool:
            self.maxpool = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_maxpool:
            x = self.maxpool(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(LinearBlock, self).__init__()
        self.linear1 = nn.Linear(in_channel, mid_channel)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(mid_channel, out_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)

        return x

class CNN(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.2):
        super(CNN, self).__init__()

        self.layers = []
        self.input_size = []

        layer = CNNBlock(in_channel, 64, 64)
        self.layers.append(layer)
        layer = CNNBlock(64, 128, 128)
        self.layers.append(layer)
        layer = CNNBlock(128, 256, 256)
        self.layers.append(layer)
        layer = CNNBlock(256, 512, 512, use_maxpool=True)
        self.layers.append(layer)

        layer = nn.Dropout3d(dropout)
        self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

        self.linear = LinearBlock(24576, 512, out_channel)

    def forward(self, x):
        if not self.input_size:
            input_size = []
            input_size.append(x.size())
            self.input_size = input_size

        for layer in self.layers:
            x = layer(x)

        x = x.view(-1, 24576)
        x = self.linear(x)

        return x

if __name__ == "__main__":
    cnn = CNN(1, 1)
    input_shape = [1, 1, 116, 132, 132]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn.to(device)


    dummy = torch.rand(input_shape).to(device)
    print("Input size : ", dummy.size())
    output = cnn(dummy)
    print("Output size : ", output.size())
    print("Input_size : ", cnn.input_size)











