import SimpleITK as sitk
import numpy as np
from pathlib import Path
import torch

def separateData(dataset_path, criteria, phase): 
    dataset = []
    for number in criteria[phase]:
        data_path = Path(dataset_path) / ("case_" + number) 

        image_list = data_path.glob("image*")
        label_list = data_path.glob("label*")
        
        image_list = sorted(image_list)
        label_list = sorted(label_list)

        for img, lab in zip(image_list, label_list):
            dataset.append((str(img), str(lab)))

    return dataset




