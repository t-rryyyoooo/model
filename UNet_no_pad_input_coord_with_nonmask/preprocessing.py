import os
import SimpleITK as sitk
import numpy as np
from random import randrange

class Compose(object):
    def __init__(self, transforms):
        """ transform input_image and target_image.

        Parameters: 
            transforms: transform (class list) -- List of classes for transforming input_image and target_image. [ex] [ReadImage(), GetArrayFromImage()]

        Note: The below method must be defined in classes belong to transforms.
        def __call__(self, input_image_or_list, target_image):
            Paramters: 
                input_image_or_lust -- an image or image list.
                target_image        -- an image
            [processing]
            return input_image, target_image
        """

        self.transforms = transforms

    def __call__(self, input_image_or_list, target_image):
        for transform in self.transforms:
            input_image_or_list, target_image = transform(input_image_or_list, target_image)
        return input_image_or_list, target_image

class MakeLabelOnehot(object):
    def __init__(self, channel_location="first", num_class=14):
        if channel_location not in ["first", "last"]:
            raise NotImplementedError("{} is not supported.".format(channel_location))

        self.channel_location = channel_location
        self.num_class        = num_class

    def __call__(self, input_array, target_array):
        s = list(range(target_array.ndim))
        onehot_target_array = np.eye(self.num_class)[target_array]
        if self.channel_location == "first":
            onehot_target_array = onehot_target_array.transpose([target_array.ndim] + s)

        return input_array, onehot_target_array

class LoadMultipleData(object):
    def __init__(self):
        """ Load image (image array) from image path (.npy, .mha).
     
        Returns: 
            np.ndarray, np.ndarray or [np.ndarray, ...], np.ndarray
        """
        self.npy_loader = LoadNpy()
        self.mha_loader = LoadMha()


    def __call__(self, input_file_or_list, target_file):
        """ Identity the file extension and load image from image path.
        
        Note: This method works on condition that the input_file extension and target_file extension is equal.
        """
        if isinstance(input_file_or_list, str):
            _, ext = os.path.splitext(input_file_or_list)# ext: extension
        else:
            _, ext = os.path.splitext(input_file_or_list[0])# ext: extension
        if ext == ".npy":
            input_image_array_or_list, target_image_array = self.npy_loader(
                                                        input_file_or_list,
                                                        target_file
                                                        )

        elif ext in [".mha", ".gz"]:
            input_image_array_or_list, target_image_array = self.mha_loader(
                                                        input_file_or_list, 
                                                        target_file
                                                        )

        else:
            raise NotImplementedError("This file extension [{}] is not supperted.".format(ext))

        return input_image_array_or_list, target_image_array


class LoadNpy(object):
    def __init__(self):
        """ Load image array from image path (.npy). 
        
        Returns: 
            np.ndarray, np.ndarray or [np.ndarray, ...], np.ndarray
        """

    def __call__(self, input_file_or_list, target_file):
        target_image_array = np.load(target_file)

        if isinstance(input_file_or_list, str):
            input_file_list = [input_file_or_list]
        else:
            input_file_list = input_file_or_list

        input_image_array_list = []
        for input_file in input_file_list:
            input_image_array = np.load(input_file)

            input_image_array_list.append(input_image_array)

        if len(input_image_array_list) == 1:
            return input_image_array_list[0], target_image_array

        else:
            return input_image_array_list, target_image_array

class LoadMha(object):
    def __init__(self):
        """ Load image array from image path (.mha). 
        
        Returns: 
            np.ndarray, np.ndarray or [np.ndarray, ...], np.ndarray
        """

    def __call__(self, input_file_or_list, target_file):
        target_image       = sitk.ReadImage(target_file)
        target_image_array = sitk.GetArrayFromImage(target_image)

        if isinstance(input_file_or_list, str):
            input_file_list = [input_file_or_list]
        else:
            input_file_list = input_file_or_list

        input_image_array_list = []
        for input_file in input_file_list:
            input_image       = sitk.ReadImage(input_file)
            input_image_array = sitk.GetArrayFromImage(input_image)

            input_image_array_list.append(input_image_array)

        if len(input_image_array_list) == 1:
            return input_image_array_list[0], target_image_array
        else:
            return input_image_array_list, target_image_array

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

    def __call__(self, input_image_array_or_list, target_image_array):
        translated_target_image_array = self.addNDim(target_image_array, self.target_ndim, self.direction)

        if isinstance(input_image_array_or_list, np.ndarray):
            input_image_array_list = [input_image_array_or_list]
            input_ndim             = [self.input_ndim]

        else:
            input_image_array_list = input_image_array_or_list
            if isinstance(self.input_ndim, int):
                input_ndim = [self.input_ndim] * len(input_image_array_list)
            else:
                input_ndim = self.input_ndim

        assert len(input_image_array_list) == len(input_ndim)

        translated_image_array_list = []
        for ndim, input_image_array in zip(input_ndim, input_image_array_list):
            translated_image_array = self.addNDim(input_image_array, ndim, self.direction)
        
            translated_image_array_list.append(translated_image_array)

        if len(translated_image_array_list) == 1:
            return translated_image_array_list[0], target_image_array
        else:
            return translated_image_array_list, translated_target_image_array

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
    def __init__(self, input_min_value=-300.0, input_max_value=300.0, target_min_value=-300, target_max_value=300):
        """ Apply the following fomula to each voxel (min max scaling).

        V_new = (V_org - min_value) / (max_value- min_value)

            min_value (float) -- Refer above.
            max_value (float) -- Refer above.
        """
        self.input_min_value  = input_min_value
        self.input_max_value  = input_max_value
        self.target_min_value = target_min_value
        self.target_max_value = target_max_value

    def __call__(self, input_image_array_or_list, target_image_array: np.ndarray):
        translated_target_image_array = self.standardize(target_image_array, self.target_min_value, self.target_max_value)

        if isinstance(input_image_array_or_list, np.ndarray):
            input_image_array_list = [input_image_array_or_list]
            input_min_value        = [self.input_min_value]
            input_max_value        = [self.input_max_value]

        else:
            input_image_array_list = input_image_array_or_list
            if isinstance(self.input_min_value, int):
                input_min_value = [self.input_min_value] * len(input_image_array_list)
            else:
                input_min_value = self.input_min_value

            if isinstance(self.input_max_value, int):
                input_max_value = [self.input_max_value] * len(input_image_array_list)
            else:
                input_max_value = self.input_max_value

        assert len(input_image_array_list) == len(input_min_value) == len(input_max_value)

        translated_image_array_list = []
        for input_image_array, min_value, max_value in zip(input_image_array_list, input_min_value, input_max_value):
            translated_image_array  = self.standardize(input_image_array, min_value, max_value)

            translated_image_array_list.append(translated_image_array)


        if len(translated_image_array_list) == 1:
            return translated_image_array_list[0], translated_target_image_array
        else:
            return translated_image_array_list, translated_target_image_array

    def standardize(self, image_array, min_value, max_value):
        image_array = image_array.clip(min=min_value, max=max_value)
        image_array = (image_array - min_value) / (max_value - min_value)

        return image_array

class Clip(object):
    def __init__(self, clip_size=[256,256]):
        self.clip_size = clip_size

    def __call__(self, input_image_array: np.ndarray, target_image_array: np.ndarray):
        assert len(self.clip_size) == input_image_array.ndim == target_image_array.ndim
        assert input_image_array.shape == target_image_array.shape
        start_index = [randrange(0, s - c) for s, c in zip(input_image_array.shape, self.clip_size)]

        input_image_array  = self.clip(input_image_array, start_index)
        target_image_array = self.clip(target_image_array, start_index)

        return input_image_array, target_image_array

    def clip(self, image_array, start_index):
        slices = []
        for si, cs in zip(start_index, self.clip_size):
            s = slice(si, si + cs)
            slices.append(s)

        slices = tuple(slices)
        print(slices)
        clipped_image_array = image_array[slices]

        return clipped_image_array

