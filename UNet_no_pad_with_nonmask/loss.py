from torch import nn
import torch

class WeightedCategoricalCrossEntropy(nn.Module):
    def __init__(self, weighted=False):
        super(WeightedCategoricalCrossEntropy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weighted = weighted 

    def forward(self, pred, true):
        """ 
        onehot
        pred, true shape : B * C * D * H * W
        """
        
        true = true.to(self.device)
        if self.weighted:
            result = torch.sum(true, dim=[0, 2, 3, 4])
        else:
            result = torch.sum(true, dim=[0, 1, 2, 3, 4])
        
        result_f = torch.pow(result, 1./3.)
        weight = result_f / torch.sum(result_f)
        if self.weighted:
            weight = weight[None, ...]
            while weight.ndim < true.ndim:
                weight = weight[..., None]
        
        output = ((-1) * torch.sum(1 / (weight + 10**-9) * true * torch.log(pred + 10**-9), axis=1))

        output = output.mean()

        return output
