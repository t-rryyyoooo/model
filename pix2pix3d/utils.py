import functools
from torch import nn
import torch
from torch.nn import init
from pathlib import Path
from .generator import UNetGenerator
from .discriminator import NLayerDiscriminator

def separateData(dataset_path, criteria, phase, input_name="input*", target_name="target*"): 
    """ Extract data path based on criteria.

    Paramters: 
        dataset_path (str) -- the directory path. This directory structure must be below because we search data based on patient_id (case_00) and file name (input_, target_).

            - dataset_path
                - case_00
                    - input_0000.npy
                    - target_0000.npy
                    
                - case_01

        criteria  (dict)   -- The dict which determines which patient is assigned to train or val. Example is below.

            {
            "train" : ["00", "01", "02"],
            "val"   : ["03", "04", "05"]
            }

        phase (str)        -- determines which phase of patients to extract. train/val (depends on criteria's key)

        input_name (str)   -- for searching input image. It must be put * at the end. [ex] "input*"
        target_name (str)  -- for searching target image. It must be put *
        at the end [ex] "target*"

    Returns: 
        paired dataset
            [
            (input_0000.npy, target_0000.npy),
            (input_0001.npy, target_0001.npy)
            ]
    """
    dataset = []
    for number in criteria[phase]:
        data_path = Path(dataset_path) / ("case_" + number) 

        input_list  = data_path.glob(input_name)
        target_list = data_path.glob(target_name)
        
        input_list  = sorted(input_list)
        target_list = sorted(target_list)

        for input_file, target_file in zip(input_list, target_list):
            dataset.append((str(input_file), str(target_file)))

    return dataset


def defineG(input_ch=1, output_ch=1, ngf=64, G_name="unet_256", norm="batch", use_dropout=False, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """ Create and return a generator.

    Parameters:
        input_ch (int)     -- the number of channels in input images
        output_ch (int)    -- the number of channels in output images
        ndf (int)          -- the number of filters in last conv layer
        G_name (str)       -- the generator name: unet_256 | unet_128
        norm (str)         -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- The name of our initialization method.
        init_gain (float)  -- Scaling factor for normal.
        gpu_ids (int list) -- which GPUs the network runs on: 0,1,2
    """

    net = None
    norm_layer = getNormLayer(norm_type=norm)

    if G_name == "unet_256":
        net = UNetGenerator(input_ch, output_ch, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    elif G_name == "unet_512":
        net = UNetGenerator(input_ch, output_ch, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    elif G_name == "unet_128":
        net = UNetGenerator(input_ch, output_ch, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    elif G_name == "unet_64":
        net = UNetGenerator(input_ch, output_ch, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)


    else:
        raise NotImplementedError("Generator model name [{}] is not recognized".format(G_name))

    return initNet(net, init_type, init_gain, gpu_ids)


def defineD(input_ch=2, ndf=64, D_name="PatchGAN", n_layers=3, norm="batch", init_type="normal", init_gain=0.02, gpu_ids=[]):
    """ Returns a discriminator.
    Parametes:
        input_ch (int)      -- the number of channels in input images
        ndf (int)           -- the number of filters in the first conv layer.
        D_name (str)        -- the architecturer's name: PatchGAN.
                               If you use any other D, add D_name.
        n_layer (int)       -- the number of conv layers in discriminator.
        norm (str)          -- the type of normalization layers used in the network.
        init_type (str)     -- the name of the initialization method.
        init_gain (float)   -- scaling factor for normal.
        gpu_ids (int list)  -- which GPUS the network runs on: 0,1,2
    """

    net = None
    norm_layer = getNormLayer(norm_type=norm)
    if D_name == "PatchGAN":
        net = NLayerDiscriminator(input_ch, ndf, n_layers=3, norm_layer=norm_layer)
    elif D_name == "n_layers":
        net = NLayerDiscriminator(input_ch, ndf, n_layers=n_layers, norm_layer=norm_layer)
    
    else:
        raise NotImplementedError("Discriminator model name [{}] is not recognized".format(D_name))

    return initNet(net, init_type, init_gain, gpu_ids)


def getNormLayer(norm_type="instance"):
    """ Return a normalization layer.
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2s, affine=False, track_running_stats=False)

    elif norm_type == "none":
        def norm_layer(x): return Identity()

    else:
        raise NotImplementedError("Normalization layer [{}] is not found".format(norm_type))

    return norm_layer

class Identity(nn.Module):
    def forward(self, x):
        return x

def initNet(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """ Initialize and return a network.
        1. Register CPU/GPU device (with multi-GPU support).
        2. Initialize the network weights.
        
    Parameters:
        net (nn.Module)    -- The network to be initialized
        init_type (str)    -- The name of an initialization method: normal
        gain (float)       -- Scaling factor for normal.
        gpu_ids (int list) -- which GPUS the network runs on: 0,1,2
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = nn.DataParallel(net, gpu_ids)

    initWeights(net, init_type, init_gain=init_gain)

    return net

def initWeights(net, init_type="normal", init_gain=0.02):
    """ Initialize network weights.
    Parameters: 
        net (nn.Module)   -- network to be initialized
        init_type (str)   -- The name of an initialization method: normal 
        init_gain (float) -- Scaling factor for normal.
    normal is used in the original pix2pix paper.
    """

    def initFunc(m): # define the initialization function
        class_name = m.__class__.__name__
        if hasattr(m, "weight") and (class_name.find("Conv") != -1 or class_name.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)

            else:
                raise NotImplementedError("Initialization method [{}] is not implemented".format(init_type))

            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif class_name.find("BatchNorm3d") != -1: # BatchNorm layer's weight is not a matrix; only normal distribution applis.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("Initialize network with {}".format(init_type))
    net.apply(initFunc) # Apply the initialization function <initFunc>


