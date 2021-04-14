from .preprocessing import *

class Pix2PixTransform():
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
                    Clip([256, 256]),
                    MinMaxStandardize(
                        min_value = -300, 
                        max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 3,
                        target_ndim = 3
                        )
                    ]),
                "val" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        min_value = -300, 
                        max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 3,
                        target_ndim = 3
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
    t_3 = Pix2PixTransform(3, 3)
    t_4 = Pix2PixTransform(4, 4)

    input_file = "/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_01/liver_resampled.mha"
    target_file = "/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_01/liver_resampled.mha"

    i, t = t_3("train", input_file, target_file)
    print(i.shape, t.shape)
    i, t = t_3("val", input_file, target_file)
    print(i.shape, t.shape)
    i, t = t_4("train", input_file, target_file)
    print(i.shape, t.shape)
    i, t = t_4("val", input_file, target_file)
    print(i.shape, t.shape)
