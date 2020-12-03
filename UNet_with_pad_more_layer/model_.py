import torch
from torch import nn
if __name__ == "__main__":
    from model_part import CreateConvBlock, CreateUpConvBlock
else:
    from .model_part import CreateConvBlock, CreateUpConvBlock


class UNetModel(nn.Module):
    def __init__(self, in_channel, nclasses, use_bn=True, use_dropout=True, dropout=0.5):
        super(UNetModel, self).__init__()
        self.use_dropout = use_dropout

        self.contracts = []
        self.expands = []

        # 512-512-8 -> 256-256-4
        contract = CreateConvBlock(in_channel, 4, 8, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 256-256-4 -> 128-128-2
        contract = CreateConvBlock(8, 8, 16, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        #128-128-2 -> 64-64-1
        contract = CreateConvBlock(16, 16, 32, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        # 64-64-1 -> 32-32-1
        contract = CreateConvBlock(32, 32, 64, n=2, use_bn=use_bn, pooling_size=(1, 2, 2))
        self.contracts.append(contract)

        #32-32-1 -> 16-16-1
        contract = CreateConvBlock(64, 64, 128, n=2, use_bn=use_bn, pooling_size=(1, 2, 2))
        self.contracts.append(contract)

        # 16-16-1 -> 8-8-1
        contract = CreateConvBlock(128, 128, 256, n=2, use_bn=use_bn, pooling_size=(1, 2, 2))
        self.contracts.append(contract)

        # 8-8-1 -> 4-4-1
        contract = CreateConvBlock(256, 256, 512, n=2, use_bn=use_bn, pooling_size=(1, 2, 2))
        self.contracts.append(contract)

        # 4-4-1
        self.lastContract = CreateConvBlock(512, 512, 1024, n=2, use_bn=use_bn, apply_pooling=False)

        self.contracts = nn.ModuleList(self.contracts)

        if use_dropout:
            self.dropout = nn.Dropout(dropout)

        #4-4-1 -> 8-8-1
        expand = CreateUpConvBlock(1024, 512, 512, 512, n=2, use_bn=use_bn, upsampling_size=(1, 2, 2))
        self.expands.append(expand)

        # 8-8-1 -> 16-16-1
        expand = CreateUpConvBlock(512, 256, 256, 256, n=2, use_bn=use_bn, upsampling_size=(1, 2, 2))
        self.expands.append(expand)

        # 16-16-1 -> 32-32-1
        expand = CreateUpConvBlock(256, 128, 128, 128, n=2, use_bn=use_bn, upsampling_size=(1, 2, 2))
        self.expands.append(expand)
         
        # 32-32-1 -> 64-64-1
        expand = CreateUpConvBlock(128, 64, 64, 64, n=2, use_bn=use_bn, upsampling_size=(1, 2, 2))
        self.expands.append(expand)

        # 64-64-1 -> 128-128-32
        expand = CreateUpConvBlock(64, 32, 32, 32, n=2, use_bn=use_bn)
        self.expands.append(expand)

        # 128-128-2 -> 256-256-4
        expand = CreateUpConvBlock(32, 16, 16, 16, n=2, use_bn=use_bn)
        self.expands.append(expand)

        # 256-256-4 -> 512-512-8
        expand = CreateUpConvBlock(16, 8, 8, 8, n=2, use_bn=use_bn)
        self.expands.append(expand)

        self.expands = nn.ModuleList(self.expands)

        self.segmentation = nn.Conv3d(8, nclasses, (1, 1, 1), stride=1, dilation=1, padding=(0, 0, 0))

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
    net_shape = (2, 1, 512, 512, 8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    #from torchsummary import summary
    #a = summary(model, net_shape)

    dummy_img = torch.rand(net_shape).to(device)
    print("input: ", net_shape)

    #output = model.forwardWithoutSegmentation(dummy_img)
    output = model(dummy_img)
    print('output:', output.size())
