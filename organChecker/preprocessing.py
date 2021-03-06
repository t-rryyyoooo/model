import os
import SimpleITK as sitk
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        """ transform input_image and target_image.

        Parameters: 
            transforms: transform (class list) -- List of classes for transforming input_image and target_image. [ex] [ReadImage(), GetArrayFromImage()]

        Note: The below method must be defined in classes belong to transforms.
        def __call__(self, input_image, target_image):
            [processing]
            return input_image, target_image
        """

        self.transforms = transforms

    def __call__(self, input_image, target_image):
        for transform in self.transforms:
            input_image, target_image = transform(input_image, target_image)
        return input_image, target_image

class TransformSegToOrganExist(object):
    """ Transform a segmentaiton image to a label whether organs exist in input image array in image array fed to a model. """
    def __call__(self, input_array: np.ndarray, target_array: np.ndarray):
        organ_existence = int((target_array > 0).any())

        return input_array, organ_existence

class AdjustDimensionality(object):
    def __init__(self, input_ndim=3, target_ndim=3, direction="head"):
        """ Adjust an image dimensionality for feeding it to model.

        Paratemters: 
            input_ndim (int)  -- the number of dimensions of the input image
            target_ndim (int) -- the number of dimensions of the target image
            direction (str)        -- Direction to add the dimension [head / tail]. Head means [np.newaxis, ...], tail means [..., np.newaxis].
        """

        self.input_ndim  = input_ndim
        self.target_ndim = target_ndim
        if direction in ["head",  "tail"]:
            self.direction = direction
        else:
            raise NotImplementedError("This argument [{}] is not supperted.".format(direction))

    def __call__(self, input_image_array: np.ndarray, target_image_array: np.ndarray):
        input_image_array  = self.addNDim(input_image_array, self.input_ndim, self.direction)
        target_image_array = self.addNDim(target_image_array, self.target_ndim, self.direction)

        return input_image_array, target_image_array

    def addNDim(self, image_array, ndim, direction):
        """ Add the number of dimensions of image_array to ndim.

        Parameters: 
            image_array (np.array) -- image array.
            ndim (int)             -- Desired number of dimensions
            direction (str)        -- Direction to add the dimension [head / tail]. Head means [np.newaxis, ...], tail means [..., np.newaxis].

        Returns: 
            image array added the number of dimensions to ndim.
        """
        while image_array.ndim < ndim:
            if direction == "head":
                image_array = image_array[np.newaxis, ...]
            elif direction == "tail":
                image_array = image_array[..., np.newaxis]

        return image_array

class MinMaxStandardize(object):
    def __init__(self, min_value=-300.0, max_value=300.0):
        """ Apply the following fomula to each voxel (min max scaling).

        V_new = (V_org - min_value) / (max_value- min_value)

            min_value (float) -- Refer above.
            max_value (float) -- Refer above.
        """
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, input_image_array: np.ndarray, target_image_array: np.ndarray):
        input_image_array  = self.standardize(input_image_array)

        return input_image_array, target_image_array

    def standardize(self, image_array):
        image_array = image_array.clip(min=self.min_value, max=self.max_value)
        image_array = (image_array - self.min_value) / (self.max_value - self.min_value)

        return image_array

class LoadNpy(object):
   
    def __init__(self):
        """ Load image array from image path (.npy). 
        
        Returns: 
            np.ndarray, np.ndarray
        """

    def __call__(self, input_file, target_file):
        input_image_array  = np.load(input_file)
        target_image_array = np.load(target_file)

        return input_image_array, target_image_array

class LoadMha(object):
    def __init__(self):
        """ Load image array from image path (.mha). 
        
        Returns: 
            np.ndarray, np.ndarray
        """

    def __call__(self, input_file, target_file):
        input_image  = sitk.ReadImage(input_file)
        target_image = sitk.ReadImage(target_file)
        input_image_array  = sitk.GetArrayFromImage(input_image)
        target_image_array = sitk.GetArrayFromImage(target_image)

        return input_image_array, target_image_array


class LoadMultipleData(object):
    def __init__(self):
        """ Load image (image array) from image path (.npy, .mha).
     
        Returns: 
            np.ndarray, np.ndarray
        """
        self.npy_loader = LoadNpy()
        self.mha_loader = LoadMha()


    def __call__(self, input_file, target_file):
        """ Identity the file extension and load image from image path.
        
        Note: This method works on condition that the input_file extension and target_file extension is equal.
        """
        _, ext = os.path.splitext(input_file)# ext: extension
        if ext == ".npy":
            input_image_array, target_image_array = self.npy_loader(
                                                        input_file,
                                                        target_file
                                                        )

        elif ext in [".mha", ".gz"]:
            input_image_array, target_image_array = self.mha_loader(
                                                        input_file, 
                                                        target_file
                                                        )

        else:
            raise NotImplementedError("This file extension [{}] is not supperted.".format(ext))

        return input_image_array, target_image_array

