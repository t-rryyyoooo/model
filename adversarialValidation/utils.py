import SimpleITK as sitk
import numpy as np
from pathlib import Path
import torch

class DICE():
    def __init__(self, num_class, device):
        self.num_class = num_class
        self.device = device
        """
        Required : not onehot (after argmax)
        ex : [[0,1], [2,5],[10,11]]
        """

    def compute(self, true, pred):
        eps = 10**-9
        assert true.size() == pred.size()
        
        true.to(self.device)
        true.to(self.device)

        
        intersection = (true * pred).sum()
        union = (true * true).sum() + (pred * pred).sum()
        dice = (2. * intersection) / (union + eps)
        """
        intersection = (true == pred).sum()
        union = (true != 0).sum() + (pred != 0).sum()
        dice = 2. * (intersection + eps) / (union + eps)
        """
        
        return dice

    def computePerClass(self, true, pred):
        DICE = []
        for x in range(self.num_class):
            true_part = (true == x).int()
            pred_part = (pred == x).int()
            """
            true_part = true[..., x]
            pred_part = pred[..., x]
            """
            dice = self.compute(true_part, pred_part)
            DICE.append(dice)

        return DICE




