from .preprocessing import *

class OrganCheckerTransform():
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
                    TransformSegToOrganExist(),
                    MinMaxStandardize(
                        min_value = -300, 
                        max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 3,
                        target_ndim = 1
                        )
                    ]),
                "val" : Compose([
                    LoadMultipleData(),
                    TransformSegToOrganExist(),
                    MinMaxStandardize(
                        min_value = -300, 
                        max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 3,
                        target_ndim = 1
                        )
                    ]),
                "test" : Compose([
                    MinMaxStandardize(
                        min_value = -300, 
                        max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 1
                        )
                    ])
                }

    def __call__(self, phase, input_image, target_image):#phase: train/val
        return self.transforms[phase](input_image, target_image)
