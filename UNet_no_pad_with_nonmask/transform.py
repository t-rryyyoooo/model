from .preprocessing import *

class UNetTransform():
    def __init__(self):
        self.transforms = {
                "train" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        input_min_value    = -300,
                        input_max_value    = 300,
                        standardize_target = False
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 3
                        )
                    ]), 

                "val" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        input_min_value    = -300,
                        input_max_value    = 300,
                        standardize_target = False
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 3
                        )
                    ]),
                "test" : Compose([
                    GetArrayFromImages(),
                    MinMaxStandardize(
                        input_min_value    = -300,
                        input_max_value    = 300,
                        standardize_target = False
                        ),
                    AdjustDimensionality(
                        input_ndim  = 5,
                        target_ndim = 3
                        )
                    ])
                }

    def __call__(self, phase, image, label):

        return self.transforms[phase](image, label)


