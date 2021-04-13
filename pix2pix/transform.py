from .preprocessing import *

class Pix2PixTransform():
    def __init__(self, clip_size=[256, 256], min_value=-300., max_value=300., input_ndim=3, target_ndim=3):
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
                    Clip(clip_size),
                    MinMaxStandardize(
                        min_value = min_value, 
                        max_value = max_value
                        ),
                    AdjustDimensionality(
                        input_ndim  = input_ndim,
                        target_ndim = target_ndim
                        )
                    ]),
                "val" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        min_value = min_value, 
                        max_value = max_value
                        ),
                    AdjustDimensionality(
                        input_ndim  = input_ndim,
                        target_ndim = target_ndim
                        )
                    ]),
                "test" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        min_value = min_value, 
                        max_value = max_value
                        ),
                    AdjustDimensionality(
                        input_ndim  = input_ndim,
                        target_ndim = target_ndim
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
