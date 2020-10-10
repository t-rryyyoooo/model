import SimpleITK as sitk
import numpy as np
from .utils import *
from random import randint

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)

        return image

class ReadImage(object):
    def __call__(self, image_file):
        image = sitk.ReadImage(image_file)

        return image

class GetArrayFromImage(object):
    def __call__(self, image):
        image_array = sitk.GetArrayFromImage(image)

        return image_array



