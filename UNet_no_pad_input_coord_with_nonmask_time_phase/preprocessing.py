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

class StackImages(object):
    def __init__(self, target_numbers=None):
        """ Stack image arrays in channel direction. 
        [ex] array1 shape : [2,3], array2 shape : [2,3] -> returned array : [2, 2, 3]
        
        Parameters: 
            target_numbers (list) -- The index of images to be stacked in image_array_list.
            [ex] target_numbers = [0,1] -> np.stack([image_array_list[0], image_array_list[1]])
        """
        self.target_numbers = target_numbers

    def __call__(self, image_array_list, target_array):
        indices = np.arange(len(image_array_list))
        if self.target_numbers is None:
            target_numbers = indices
        else:
            target_numbers = self.target_numbers

        rest_indices = np.delete(indices, target_numbers)

        concated_image_array = np.stack(np.array(image_array_list)[target_numbers])

        returned_array_list = [concated_image_array] + list(np.array(image_array_list)[rest_indices])

        if len(returned_array_list) == 1:
            return returned_array_list[0], target_array
        else:
            return returned_array_list, target_array

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
            raise NotImplementedError("{} mode is not supported.".format(mode))
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

class ClipValues(object):
    def __init__(self, input_min_value=-300.0, input_max_value=300.0, target_min_value=-300, target_max_value=300):
        """ Clip values in image array from min_value to max_value.
        if You don't want to clip values, set None.

        """
        self.input_min_value  = input_min_value
        self.input_max_value  = input_max_value
        self.target_min_value = target_min_value
        self.target_max_value = target_max_value

    def __call__(self, input_array_or_list, target_array: np.ndarray):
        if self.target_min_value is None and self.target_max_value is None:
            clipped_target_array = target_array
        else:
            clipped_target_array = target_array.clip(
                                    min = self.target_min_value,
                                    max = self.target_max_value
                                    )

        if isinstance(input_array_or_list, np.ndarray):
            input_array_list = [input_array_or_list]
            input_min_value        = [self.input_min_value]
            input_max_value        = [self.input_max_value]

        else:
            input_array_list = input_array_or_list
            if isinstance(self.input_min_value, int):
                input_min_value = [self.input_min_value] * len(input_array_list)
            else:
                input_min_value = self.input_min_value

            if isinstance(self.input_max_value, int):
                input_max_value = [self.input_max_value] * len(input_array_list)
            else:
                input_max_value = self.input_max_value

        assert len(input_array_list) == len(input_min_value) == len(input_max_value)

        clipped_input_array_list = []
        for input_array, min_value, max_value in zip(input_array_list, input_min_value, input_max_value):
            if min_value is None and max_value is None:
                clipped_input_array = input_array
            else:
                clipped_input_array = input_array.clip(
                                            min = min_value,
                                            max = max_value
                                            )

            clipped_input_array_list.append(clipped_input_array)


        if len(clipped_input_array_list) == 1:
            return clipped_input_array_list[0], clipped_target_array
        else:
            return clipped_input_array_list, clipped_target_array


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
        clipped_image_array = image_array[slices]

        return clipped_image_array

if __name__ == "__main__":
    image_1 = "/home/vmlab/Desktop/data/Abdomen/case_00/imaging_resampled_2_uni.nii.gz"
    image_2 = "/home/vmlab/Desktop/data/Abdomen/case_00/translate_resampled_2_uni.mha"
    image_3 = "/home/vmlab/Desktop/data/Abdomen/case_00/segmentation_resampled_2.nii.gz"

    array_list = [image_1, image_2, image_1]
    c = Compose([
            LoadMultipleData(),
            StackImages([0,2]),
            ClipValues(
                input_min_value = [-140,None],
                input_max_value = [140, None],
                target_min_value = None,
                target_max_value = None
                )
            ])

    a, s = c(array_list, image_3)

    print(a[0].shape, a[0].min(), a[0].max())
    print(a[1].shape, a[1].min(), a[1].max())
    print(s.shape, s.min(), s.max())


