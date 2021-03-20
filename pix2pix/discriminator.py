import functools
from torch import nn

class NLayerDiscriminator(nn.Module):
    """ Defines a PatchGAN discriminator. """
    def __init__(self, input_ch, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """ Construct a PatchGAN discriminator.
        Parameters:
            input_ch (int) -- the number of channels in input images
            ndf (int)      -- the number of filters in the last (first) conv layer
            n_layers (int) -- the number of layers in the discriminator
            norm_layer     -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [nn.Conv2d(input_ch, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1 # ndf * nf_mult means the number of output feature maps for nn.Conv2d. Max: ndf * 8
        nf_mult_prev = 1# ndf * nf_mult_prev means the number of input feature map for nn.Conv2d. Max: ndf * 8
        for n in range(1, n_layers): # Gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                    ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
                ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)] # Output 1 channel prediction map

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    input_ch = 64
    input_size = 70
    nd = NLayerDiscriminator(input_ch)

    import torch
    x = torch.Tensor(1, 64, input_size, input_size)
    print(x.size())
    pred = nd.forward(x)
    print(pred.size())

