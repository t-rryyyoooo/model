import SimpleITK as sitk
import numpy as np
from pathlib import Path
import torch
import random

def setMeta(to_image, ref_image, direction=None, origin=None, spacing=None):
    if direction is None:
        direction = ref_image.GetDirection()

    if origin is None:
        origin = ref_image.GetOrigin()

    if spacing is None:
        spacing = ref_image.GetSpacing()

    to_image.SetDirection(direction)
    to_image.SetOrigin(origin)
    to_image.SetSpacing(spacing)

    return to_image

def separateDataWithNonMask(dataset_mask_path, dataset_nonmask_path, criteria, phase, rate={"train":{"mask":1.0, "nonmask":0.07}, "val":{"mask":0.1, "nonmask":0.1}}):
    """
    rate = {
        'train' : {'mask' : 1.0, 'nonmask' : 0.07},
        'val' : {'mask : 0.1, 'nonmask' : 0.04}
        }
    """

    dataset_mask_path = separateData(dataset_mask_path, criteria, phase, rate[phase]["mask"])
    dataset_nonmask_path = separateData(dataset_nonmask_path, criteria, phase, rate[phase]["nonmask"])

    dataset = dataset_mask_path + dataset_nonmask_path
    print("phase:", phase)
    print("The number of mask dataset:", len(dataset_mask_path))
    print("The number of nonmask dataset:", len(dataset_nonmask_path))
    print("The number of all of dataset:", len(dataset))

    return dataset

def separateData(dataset_path, criteria, phase, rate=1.0): 
    dataset = []
    random.shuffle(criteria[phase])
    for number in criteria[phase][:int(len(criteria[phase]) * rate)]:
        data_path = Path(dataset_path) / ("case_" + number) 

        image_list = data_path.glob("image*")
        label_list = data_path.glob("label*")
        
        image_list = sorted(image_list)
        label_list = sorted(label_list)

        assert len(image_list) == len(label_list)
        for img, lab in zip(image_list, label_list):
            dataset.append([str(img), str(lab)])

    return dataset


def makeAffineParameters(image, translate, rotate, shear, scale):
    dimension = image.GetDimension()
    translation = np.random.uniform(-translate, translate, dimension)
    rotation = np.radians(np.random.uniform(-rotate, rotate))
    shear = np.random.uniform(-shear, shear, 2)
    scale = np.random.uniform(1 - scale, 1 + scale)
    center = (np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2)[::-1]

    return [translation, rotation, scale, shear, center]

def makeAffineMatrix(translate, rotate, scale, shear, center):
    a = sitk.AffineTransform(3)

    a.SetCenter(center)
    a.Rotate(1, 0, rotate)
    a.Shear(1, 0, shear[0])
    a.Shear(0, 1, shear[1])
    a.Scale((scale, scale, 1))
    a.Translate(translate)

    return a

def transforming(image, bspline, affine, interpolator, minval):
    # B-spline transformation
    if bspline is not None:
        transformed_b = sitk.Resample(image, bspline, interpolator, minval)

    # Affine transformation
        transformed_a = sitk.Resample(transformed_b, affine, interpolator, minval)

    else:
        transformed_a = sitk.Resample(image, affine, interpolator, minval)

    return transformed_a

def getMinimumValue(image):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(image)
    return minmax.GetMinimum()

def cropping3D(image, crop_z, crop_x, crop_y):
    """
    image : only 5D tensor
    
    """
    size_z, size_x, size_y = image.size()[2:]
    crop_z = slice(crop_z[0], size_z - crop_z[1])
    crop_x = slice(crop_x[0], size_x - crop_x[1])
    crop_y = slice(crop_y[0], size_y - crop_y[1])
    
    cropped_image = image[..., crop_z, crop_x, crop_y]
    
    return cropped_image

class DICEPerClass():
    def __init__(self):
        super(DICEPerClass, self).__init__()

    def __call__(self, pred: torch.Tensor, true: torch.Tensor, smooth=1.0):

        axis = list(range(pred.ndim))
        del axis[0:2]
        intersection = (pred * true).sum(axis)

        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        true = true.contiguous().view(true.shape[0], true.shape[1], -1)

        pred_sum = pred.sum((-1,))
        true_sum = true.sum((-1,))

        dice_per_class = ((2. * intersection + smooth) / (pred_sum + true_sum + smooth)).mean(0)

        return dice_per_class
