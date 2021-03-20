from torch import nn
import functools

class UNetGenerator(nn.Module):
    def __init__(self, input_ch, output_ch, num_down, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """ Constract a U-Net generator.
        Parameters:
            input_ch (int)  -- the number of channels in input images
            output_ch (int) -- the number of channels in output images
            num_down (int)  -- the number of downsaplings in UNet. 
                               ex) if num_down == 7, 128x128 → 1x1 at the bottoleneck (image size)
            ngf             -- the number of filters in the last conv layer
            norm_layer      -- normalization layer


        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """

        super(UNetGenerator, self).__init__()
        """ Constract U-Net structure. """

        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_ch=None, submodule=None, norm_layer=norm_layer, innermost=True) # Innermost layer

        for i in range(num_down - 5): # add intermediate layers with ngf * 8 filters. We do not increase the number of channels by more than ngf * 8.
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_ch=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        """ Gradually reduce the number of filters from ngf * 8 to ngf. """
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_ch=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_ch=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_ch=None, submodule=unet_block, norm_layer=norm_layer)

        self.model = UNetSkipConnectionBlock(output_ch, ngf, input_ch, submodule=unet_block, outermost=True, norm_layer=norm_layer) # Outermost Layer

    def forward(self, x):
        return self.model(x)


class UNetSkipConnectionBlock(nn.Module):
    """ Defines the U-Net submodule with skip connection. 
        X ---------------------identity---------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_ch, inner_ch, input_ch=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """ Constract a U-Net submodule with skip connections. 
        Parametes:
            outer_ch (int) -- the number of filters in the outer conv layer
            inner_ch (int) -- the number of filters in the inner conv layer
            input_ch (int) -- the number of channels in input images/features
            submodule (UNetSkipConnectionBlock) -- previously defined submodules
            outermost (bool) -- if this module is the outermost module
            innermost (bool) -- if this module is the innermost module
            norm_layer       -- normalization layer
            use_dropout (bool) -- Whether use dropout layers.
        """

        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_ch is None:
            input_ch = outer_ch

        down_conv = nn.Conv2d(input_ch, inner_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)# Half the image size.
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_ch)

        up_relu = nn.ReLU(True)
        up_norm = norm_layer(outer_ch)

        if outermost:
            up_conv = nn.ConvTranspose2d(inner_ch * 2, outer_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)# Double the image size.
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            up_conv = nn.ConvTranspose2d(inner_ch, outer_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up

        else:
            up_conv = nn.ConvTranspose2d(inner_ch * 2, outer_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)# Double the image size.
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)

        else:# add skip connections
            return torch.cat([x, self.model(x)], 1)

# Test
if __name__ == "__main__":
    input_ch    = 1
    output_ch   = 1
    num_down    = 8
    use_dropout = True
    device      = "cpu"

    ug = UNetGenerator(
            input_ch    = input_ch,
            output_ch   = output_ch,
            num_down    = num_down,
            use_dropout = use_dropout
            )
    ug.to(device)

    import torch
    x = torch.Tensor(1, input_ch, 2**num_down, 2**num_down) 
    print(x.size())
    pred = ug.forward(x)
    print(pred.size())







