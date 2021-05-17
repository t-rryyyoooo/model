from .preprocessing import *

class Pix2Pix3dTransform():
    def __init__(self):
        """ Define which transformation you want to to do. 
        If you want to transform images more, just add transformation class to list in Compose as below.

        Compose([
            LoadMultipleData(input_ndim, taget_ndim),
            AffineTransform(...)
            ])
        """

        self.transforms = {
                "train" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        min_value = -300, 
                        max_value = 300
                        ),
                    #ElasticTransform(),
                    #Clip([256, 256]),
                    #RandomFlip(),
                    #RandomRotate90(),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 4
                        )
                    ]),
                "val" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        min_value = -300, 
                        max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 4
                        )
                    ]),
                "test" : Compose([
                    MinMaxStandardize(
                        min_value = -300, 
                        max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 3
                        )
                    ])
                }

    def __call__(self, phase, input_image, target_image):#phase: train/val
        return self.transforms[phase](input_image, target_image)

#Test
if __name__ == "__main__":
    t_3 = Pix2Pix3dTransform()
    t_4 = Pix2Pix3dTransform()

    input_file  = "/Users/tanimotoryou/Desktop/c.mha"
    target_file = "/Users/tanimotoryou/Desktop/c.mha"


    i, t = t_3("train", input_file, target_file)
    sitk.WriteImage(sitk.GetImageFromArray(i), "/Users/tanimotoryou/Desktop/d.mha")
