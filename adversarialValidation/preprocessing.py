import SimpleITK as sitk
import numpy as np

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
        if image_array.ndim != 4:
            image_array = image_array[np.newaxis, ...]

        return image_array



