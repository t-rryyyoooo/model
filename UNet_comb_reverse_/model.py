import torch
from torch import nn
from torchsummary import summary
if __name__ == "__main__":
    from utils import cropping3D
    from model_part import CreateConvBlock, CreateUpConvBlock, CreateFinalBlock
else:
    from .utils import cropping3D
    from .model_part import CreateConvBlock, CreateUpConvBlock, CreateFinalBlock

class UNetCombReverseModel(nn.Module):
    def __init__(self, in_channel_main, in_channel_final, nclasses, use_bn=True, use_dropout=True):
        super(UNetCombReverseModel, self).__init__()
        self.use_dropout = use_dropout

        self.contracts = []
        self.expands = []

        self.contract_first = CreateConvBlock(in_channel_main, 32, 64, n=2, use_bn=use_bn)

        contract = CreateConvBlock(64, 64, 128, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(128, 128, 256, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        self.lastContract = CreateConvBlock(256, 256, 512, n=2, use_bn=use_bn, apply_pooling=False)

        self.contracts = nn.ModuleList(self.contracts)

        if use_dropout:
            self.dropout = nn.Dropout(0.5)

        expand = CreateUpConvBlock(512, 256, 256, 256, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(256, 128, 128, 128, n=2, use_bn=use_bn)
        self.expands.append(expand)
         
        self.expands = nn.ModuleList(self.expands)

        self.expand_final = CreateFinalBlock(128, 64, in_channel_final, 64, 64, n=2, use_bn=use_bn)


        self.segmentation = nn.Conv3d(64, nclasses, (1, 1, 1), stride=1, dilation=1, padding=(0, 0, 0))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_final):
        x, conv_result = self.contract_first(x)
        conv_result_first = conv_result

        conv_results = []
        for contract in self.contracts:
            x, conv_result = contract(x)
            conv_results.append(conv_result)

        conv_results = conv_results[::-1]

        x, _ = self.lastContract(x)
        if self.use_dropout:
            x = self.dropout(x)
            
        for expand, conv_result in zip(self.expands, conv_results):
            x = expand(x, conv_result)

        x = self.expand_final(x, conv_result_first, x_final)

        x = self.segmentation(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    model=UNetCombReverse(1, 64, 3)
    net_shape = (1, 1, 44+ 44*2, 44 + 44*2, 28 + 44*2)
    net_shape_final = (1, 64, 48, 48, 32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    dummy_img = torch.rand(net_shape).to(device)
    dummy_img_final = torch.rand(net_shape_final).to(device)
    print("input: ", net_shape, net_shape_final)

    output = model(dummy_img, dummy_img_final)
    net_shape = (1, 44+ 44*2, 44 + 44*2, 28 + 44*2)
    net_shape_final = (64, 48, 48, 32)
    summary(model, [net_shape, net_shape_final])
    print('output:', output.size())
