import torch
from torch import nn
if __name__ == "__main__":
    from modelPart import CreateConvBlock, CreateUpConvBlock
else:
    from .modelPart import CreateConvBlock, CreateUpConvBlock


class UNetModel(nn.Module):
    def __init__(self, in_channel, nclasses, use_bn=True, use_dropout=True, dropout=0.5):
        super(UNetModel, self).__init__()
        self.use_dropout = use_dropout

        self.contracts = []
        self.expands = []

        # 512 -> 256
        contract = CreateConvBlock(in_channel, 64, 64, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 256 -> 128
        contract = CreateConvBlock(64, 128, 128, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 128 -> 64
        contract = CreateConvBlock(128, 256, 256, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 64 -> 32
        contract = CreateConvBlock(256, 512, 512, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 32 -> 16
        contract = CreateConvBlock(512, 512, 512, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 16 -> 8
        contract = CreateConvBlock(512, 512, 512, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 8 -> 4
        contract = CreateConvBlock(512, 512, 512, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 4 -> 2
        contract = CreateConvBlock(512, 512, 512, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        self.lastContract = CreateConvBlock(512, 512, 512, n=2, use_bn=use_bn, apply_pooling=False)

        self.contracts = nn.ModuleList(self.contracts)

        if use_dropout:
            self.dropout = nn.Dropout(dropout)

        expand = CreateUpConvBlock(512, 512, 512, 512, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(512, 512, 512, 512, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(512, 512, 512, 512, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(512, 512, 512, 512, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(512, 512, 512, 512, n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(512, 256, 256, 256, n=2, use_bn=use_bn)
        self.expands.append(expand)
         
        expand = CreateUpConvBlock(256, 128, 128, 128, n=2, use_bn=use_bn)
        self.expands.append(expand)
         
        expand = CreateUpConvBlock(128, 64, 64, 64, n=2, use_bn=use_bn)
        self.expands.append(expand)

        self.expands = nn.ModuleList(self.expands)

        self.segmentation = nn.Conv2d(64, nclasses, 1, stride=1, dilation=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
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

        x = self.segmentation(x)
        x = self.softmax(x)

        return x

    def forwardWithoutSegmentation(self, x):
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


        return x


if __name__ == "__main__":
    model=UNetModel(1 ,14)
    net_shape = (1, 1, 512, 512)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    #from torchsummary import summary
    #a = summary(model, net_shape)

    dummy_img = torch.rand(net_shape).to(device)
    print("input: ", net_shape)

    #output = model.forwardWithoutSegmentation(dummy_img)
    output = model(dummy_img)
    print('output:', output.size())
