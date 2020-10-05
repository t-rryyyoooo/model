import SimpleITK as sitk
import numpy as np
from random import randint
if __name__ == "__main__":
    from utils import *
else:
    from .utils import *


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)

        return image, label

class ReadImage(object):
    def __call__(self, image_file, label_file):
        image = sitk.ReadImage(image_file)
        label = sitk.ReadImage(label_file)

        return image, label

class ReadImages(object):
    def __call__(self, image_file_list, label_file):
        image_list = []
        for image_file in image_file_list:
            image = sitk.ReadImage(image_file)
            image_list.append(image)

        label_array = np.load(label_file)

        return image_list, label_array

class GetArrayFromImages(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, image_list, label):
        image_array_list = []
        for image in image_list:
            image_array = sitk.GetArrayFromImage(image)

            if image.GetDimension() != 4:
                image_array = image_array[np.newaxis, ...]
            image_array_list.append(image_array)

        label_array = sitk.GetArrayFromImage(label).astype(int)
        
        return image_array_list, label_arry

class LoadNumpys(object):
    def __call__(self, image_file_list, label_file):
        image_array_list = []
        for image_file in image_file_list:
            image_array = np.load(image_file)
            if image_array.ndim != 4:
                image_array = image_array[np.newaxis, ...]
            image_array_list.append(image_array)

        label_array = np.load(label_file)

        return image_array_list, label_array

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


