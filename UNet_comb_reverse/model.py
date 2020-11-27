import torch
from torch import nn
from torchsummary import summary
if __name__ == "__main__":
    from model_part import CreateFinalBlock
    from model_org import UNetModel
else:
    from .model_part import CreateFinalBlock
    from .model_org import UNetModel

class UNetCombReverseModel(nn.Module):
    def __init__(self, in_channel_main, in_channel_final, nclasses, dropout=0.5, transfer_org_model=None, use_bn=True, use_dropout=True):
        super(UNetCombReverseModel, self).__init__()
        self.use_dropout = use_dropout

        if transfer_org_model is None:
            self.org_model = UNetModel(
                    in_channel = in_channel_main, 
                    nclasses = nclasses, 
                    dropout=dropout, 
                    use_bn=use_bn, 
                    use_dropout=use_dropout
                    )

        else:
            self.org_model = transfer_org_model

        self.expand_final = CreateFinalBlock(128, 64, in_channel_final, 64, 64, n=2, use_bn=use_bn)

        self.segmentation = nn.Conv3d(64, nclasses, (1, 1, 1), stride=1, dilation=1, padding=(0, 0, 0))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_final):
        print(self.org_model.contracts[0])
        x, conv_result_first = self.org_model.contracts[0](x)

        conv_results = []
        for contract in self.org_model.contracts[1:]:
            x, conv_result = contract(x)
            conv_results.append(conv_result)

        conv_results = conv_results[::-1]

        x, _ = self.org_model.lastContract(x)

        if self.use_dropout:
            x = self.org_model.dropout(x)
            
        for expand, conv_result in zip(self.org_model.expands[:-1], conv_results):
            x = expand(x, conv_result)

        x = self.expand_final(x, conv_result_first, x_final)

        x = self.segmentation(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    model=UNetCombReverseModel(1, 64, 3)

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
