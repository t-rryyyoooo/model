from torch import nn
import torch
from torch.nn import functional as F

class WeightedCategoricalCrossEntropy(nn.Module):
    def __init__(self, weighted=False):
        super(WeightedCategoricalCrossEntropy, self).__init__()
        self.weighted = weighted
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, true):
        """ 
        onehot
        pred, true shape : B * C * D * H * W
        """
        
        true = true.to(self.device)
        if self.weighted:
            result = torch.sum(true, dim=[0, 2, 3, 4])
            result_f = torch.pow(result, 1./3.)
            
            weight = result_f / torch.sum(result_f)
            weight = weight[None, ...]
            while weight.ndim < true.ndim:
                weight = weight[..., None]

        else:
            result = torch.sum(true, dim=[0, 1, 2, 3, 4])
            result_f = torch.pow(result, 1./3.)
            
            weight = result_f / torch.sum(result_f)
        
        output = (-1) * torch.sum(1 / (weight + 10**-9) * true * torch.log(pred + 10**-9), axis=1)

        output = output.mean()

        return output

class WholeDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(WholeDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DICEPerClassLoss(torch.nn.Module):
    def __init__(self):
        super(DICEPerClassLoss, self).__init__()

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor, smooth=1.0):
        """
        :param pred:
        :param teacher:
        :param smooth:
        :return:
        """
        pred, teacher = pred.float(), teacher.float()

        # batch size, classes, width and height
        axis = list(range(pred.ndim))
        del axis[0:2]

        intersection = (pred * teacher).sum(axis)
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        teacher = teacher.contiguous().view(teacher.shape[0], teacher.shape[1], -1)

        pred_sum = pred.sum((-1,))
        teacher_sum = teacher.sum((-1,))

        num_existence = ((pred_sum + teacher_sum) > 0).any(0).float().sum()

        dice_by_classes = (2. * intersection + smooth) / (pred_sum + teacher_sum + smooth)

        return (1. - dice_by_classes).sum() / num_existence 
        #return (1. - dice_by_classes).mean()#.mean((-1,)).mean((-1,))

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss

if __name__ == "__main__":
    dpcl = DICEPerClassLoss()
    import numpy as np
    a = [
            [[1,1,0,0,0],[1,0,0,0,1],[0,0,0,0,0]],
            [[0,1,1,0,1],[1,1,1,0,0],[0,0,0,0,0]],
            [[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]],
            ]

    b = [
            [[0,1,0,0,0],[1,1,1,1,1],[0,0,0,0,0]],
            [[1,0,1,0,1],[0,1,0,0,0],[0,0,0,0,0]],
            [[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]],
            ]

    a = torch.from_numpy(np.array(a, dtype=float))
    b = torch.from_numpy(np.array(b, dtype=float))

    loss = dpcl(a, b)#, smooth=0.)


