import SimpleITK as sitk
import os
import numpy as np
from .utils import *
from random import randint, uniform

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)

        return image, label

class MixImages(object):
    def __init__(min_rate=-0.3, max_rate=0.3, constant_value=0.3, mode="dynamic"):
        """ Mix voxel values of the two images.
        CT_new = (1 - alpha) * CT_0 + alpha * CT_1
        Then, alpa is determined by random.uniform(min_rate, max_rate) when mode is 'dynamic'.
        When mode is 'static', constant_value is substituted for alpha.

        Parameters: 
            min_rate (float)       -- Minimum value of mixing ratio when mode is 'dynamic'.
            max_rate (float)       -- Maximum value of mixing ratio when mode is 'dynamic'.
            constant_value (float) -- Mixing ratio when mode is 'static'
            mode (str)             -- Whether alpha is changed every time this class is called. [dynamic / static]
        """

        self.min_rate = min_rate
        self.max_rate = max_rate
        if mode not in ["dynamic", "static"]:
            raise NotImplementedError("{} mode is not supported."format(mode))
        else:
            self.mode = mode

        self.constant_value = constant_value

    def __call__(self, image_array_list, label_array):
        assert len(image_array_list) == 2
        if self.mode == "dynamic":
            alpha = uniform(self.min_rate, self.max_rate)

        elif self.mode == "static":
            alpha = self.constant_value

        mixed_image_array = (1 - alpha) * image_array_list[0] + alpha * image_array_list[1]

        return mixed_image_array, label_array

class LoadMultipleData(object):
    def __call__(self, image_file_list, label_file):
        image_array_list = []
        for image_file in image_file_list:
            _, ext = os.path.splitext(image_file)
            if ext == ".mha":
                image = sitk.ReadImage(image_file)
                image_array = sitk.GetArrayFromImage(image)
            elif ext == ".npy":
                image_array = np.load(image_file)

            while image_array.ndim !=4:
                image_array = image_array[np.newaxis, ...]

            image_array_list.append(image_array)

        _, ext = os.path.splitext(label_file)
        if ext == ".mha":
            label = sitk.ReadImage(label_file)
            label_array = sitk.GetArrayFromImage(label)
        elif ext == ".npy":
            label_array = np.load(label_file)

        return image_array_list, label_array


class LoadNumpys(object):
    def __call__(self, image_file, label_file):
        image_array = np.load(image_file)
        if image_array.ndim != 4:
            image_array = image_array[np.newaxis, ...]

        label_array = np.load(label_file)

        return image_array, label_array

class ReadImage(object):
    def __call__(self, image_file, label_file):
        image = sitk.ReadImage(image_file)
        label = sitk.ReadImage(label_file)

        return image, label

class AffineTransform(object):
    def __init__(self, translate_range, rotate_range, shear_range, scale_range, bspline=None):
        self.translate_range = translate_range
        self.rotate_range = rotate_range
        self.shear_range = shear_range
        self.scale_range = scale_range
        self.bspline = None

    def __call__(self, image, label):
        """
        image : 116 * 132 * 132
        label : 28 * 44 * 44
        """
        image_parameters = makeAffineParameters(image, self.translate_range, self.rotate_range, self.shear_range, self.scale_range)
        label_parameters = image_parameters[:]
        """ Because image and label are the different size, we must change center paramter per image. """
        label_parameters[-1] = (np.array(label.GetSize()) * np.array(label.GetSpacing()) / 2)[::-1]
        
        
        image_affine = makeAffineMatrix(*image_parameters)
        label_affine = makeAffineMatrix(*label_parameters)
        minval = getMinimumValue(image)
        transformed_image = transforming(image, self.bspline, image_affine, sitk.sitkLinear, minval)

        transformed_label = transforming(label, self.bspline, label_affine, sitk.sitkNearestNeighbor, 0)

        return transformed_image, transformed_label

class GetArrayFromImage(object):
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, image, label):
        image_array = sitk.GetArrayFromImage(image)
        label_array = sitk.GetArrayFromImage(label).astype(int)

        if image.GetDimension() != 4:
            #image_array = image_array[..., np.newaxis]
            image_array = image_array[np.newaxis, ...]

        #image_array = image_array.transpose((3, 2, 0, 1))
        #label_array = label_array.transpose((2, 0, 1))

        return image_array, label_array

class RandomFlip(object):
    def __call__(self, image, label):
        dimension = image.GetDimension()
        flip_filter = sitk.FlipImageFilter()

        flip_axes = [bool(randint(0, 1)) for _ in range(dimension)]
        flip_filter.SetFlipAxes(flip_axes)

        flipped_image = flip_filter.Execute(image)
        flipped_label = flip_filter.Execute(label)

        flipped_image = setMeta(flipped_image, image)
        flipped_label = setMeta(flipped_label, label)

        return flipped_image, flipped_label


