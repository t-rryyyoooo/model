import torch
from torch import nn
if __name__ == "__main__":
    from utils import cropping3D
    from modelPart import FirstConvBlock, CreateConvBlock, CreateUpConvBlock
else:
    from .utils import cropping3D
    from .modelPart import FirstConvBlock, CreateConvBlock, CreateUpConvBlock

from torchsummary import summary

class UNetModel(nn.Module):
    def __init__(self, in_channel_img, in_channel_coord, nclasses, dropout=0.5, use_bn=True, use_dropout=True):
        super(UNetModel, self).__init__()
        self.use_dropout = use_dropout

        self.contracts = []
        self.expands = []

        self.first_contract = FirstConvBlock(in_channel_img, in_channel_coord, 32, 64, n=2, use_bn=use_bn)

        contract = CreateConvBlock(64, 64, 128, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(128, 128, 256, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        self.lastContract = CreateConvBlock(256, 256, 512, n=2, use_bn=use_bn, apply_pooling=False)

        self.contracts = nn.ModuleList(self.contracts)

        if use_dropout:
            self.dropout = nn.Dropout(dropout)

        expand = CreateUpConvBlock(512, 256, 256, 256, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(256, 128, 128, 128, n=2, use_bn=use_bn)
        self.expands.append(expand)
         
        expand = CreateUpConvBlock(128, 64, 64, 64, n=2, use_bn=use_bn)
        self.expands.append(expand)

        self.expands = nn.ModuleList(self.expands)

        self.segmentation = nn.Conv3d(64, nclasses, (1, 1, 1), stride=1, dilation=1, padding=(0, 0, 0))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_img, x_coord):
        conv_results = []

        x, conv_result = self.first_contract(x_img, x_coord)
        conv_results.append(conv_result)
        for contract in self.contracts:
            x, conv_result = contract(x)
            conv_results.append(conv_result)

        conv_results = conv_results[::-1]

        x, _ = self.lastContract(x)
        if self.use_dropout:
            x = self.dropout(x)
            
        for expand, conv_result in zip(self.expands, conv_results):
            x = expand(x, conv_result)

        x = self.segmentation(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    model=UNetModel(1 ,3, 14)
    img_shape = (1, 1, 44+ 44*2, 44 + 44*2, 28 + 44*2)
    coord_shape = (1, 3, 44+ 44*2, 44 + 44*2, 28 + 44*2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    dummy_img = torch.rand(img_shape).to(device)
    dummy_coord = torch.rand(coord_shape).to(device)
    print("input: ", img_shape, coord_shape)

    output = model(dummy_img, dummy_coord)
    #summary(model, net_shape)
    print('output:', output.size())
