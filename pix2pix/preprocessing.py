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

class Transform(object):
    def __init__(self, input_ndim=3, target_ndim=3):
        """ Parent class for the one belong to transforms fed to Compose.

        Paratemters: 
            input_ndim (int)  -- the number of dimensions of the input image
            target_ndim (int) -- the number of dimensions of the target image
        """

        self.input_ndim  = input_ndim
        self.target_ndim = target_ndim

    def adjustNDim(self, input_image, target_image):
        input_image  = self.addNDim(input_image, self.input_ndim)
        target_image = self.addNDim(target_image, self.target_ndim)

        return input_image, target_image

    def addNDim(self, image_array, ndim):
        """ Add the number of dimensions of image_array to ndim.

        Parameters: 
            image_array (np.array) -- image array.
            ndim (int)             -- Desired number of dimensions

        Returns: 
            image array added the number of dimensions to ndim.
        """
        while image_array.ndim < ndim:
            image_array = image_array[np.newaxis, ...]

        return image_array

class LoadNpy(Transform):
   
    def __init__(self, input_ndim=3, target_ndim=3):
        """ Load image array from image path (.npy). 
        
        Paratemters: 
            input_ndim (int)  -- the number of dimensions of the input image
            target_ndim (int) -- the number of dimensions of the target image
        """
        super().__init__(input_ndim, target_ndim)

    def __call__(self, input_file, target_file):
        input_image_array  = np.load(input_file)
        target_image_array = np.load(target_file)

        input_image_array, target_image_array = super().adjustNDim(
                                                    input_image_array, 
                                                    target_image_array
                                                    )

        return input_image_array, target_image_array

class LoadMha(Transform):
    def __init__(self, input_ndim=3, target_ndim=3):
        """ Load image array from image path (.mha). 
        
        Paratemters: 
            input_ndim (int)  -- the number of dimensions of the input image
            target_ndim (int) -- the number of dimensions of the target image
        """
        super().__init__(input_ndim, target_ndim)

    def __call__(self, input_file, target_file):
        input_image  = sitk.ReadImage(input_file)
        target_image = sitk.ReadImage(target_file)
        input_image_array  = sitk.GetArrayFromImage(input_image)
        target_image_array = sitk.GetArrayFromImage(target_image)

        input_image_array, target_image_array = super().adjustNDim(
                                                    input_image_array, 
                                                    target_image_array
                                                    )

        return input_image_array, target_image_array


class LoadMultipleData(object):
    def __init__(self, input_ndim, target_ndim):
        """ Load image (image array) from image path (.npy, .mha).
     
            Paratemters: 
                input_ndim (int)  -- the number of dimensions of the input image
                target_ndim (int) -- the number of dimensions of the target image
        """
        self.npy_loader = LoadNpy(input_ndim, target_ndim)
        self.mha_loader = LoadMha(input_ndim, target_ndim)


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


# Test
if __name__ == "__main__":
    def testClass(input_file, target_file):
        multi_loader_3 = LoadMultipleData(3, 3)
        multi_loader_4 = LoadMultipleData(4, 4)
        input_image_array, target_image_array = multi_loader_3(
                                                    input_file,
                                                    target_file
                                                    )
        print("the number of dimensions: ", 3)
        print("input image array shape: ", input_image_array.shape)
        print("target image array shape: ", target_image_array.shape)

        input_image_array, target_image_array = multi_loader_4(
                                                    input_file,
                                                    target_file
                                                    )

        print("the number of dimensions: ", 4)
        print("input image array shape: ", input_image_array.shape)
        print("target image array shape: ", target_image_array.shape)

    input_file_gz  = "/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_00/imaging_resampled.nii.gz"
    target_file_gz = "/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_01/imaging_resampled.nii.gz"
    input_file_mha = "/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_00/liver_resampled.mha"
    target_file_mha = "/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_01/liver_resampled.mha"
    input_file_npy = "/Users/tanimotoryou/Desktop/test.npy"
    target_file_npy = "/Users/tanimotoryou/Desktop/test.npy"
    testClass(input_file_gz, target_file_gz)
    testClass(input_file_mha, target_file_mha)
    testClass(input_file_npy, target_file_npy)
