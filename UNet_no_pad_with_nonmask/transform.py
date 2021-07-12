from .preprocessing import *

class UNetTransform():
    def __init__(self, num_class=1):

        self.transforms = {
                "train" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        input_min_value    = -100,
                        input_max_value    = 400,
                        target_min_value   = -300,
                        target_max_value   = 300,
                        standardize_target = False,
                        ),
                    MakeLabelOnehot(
                        channel_location = "first",
                        num_class        = num_class
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 4
                        )
                    ]), 
                "val" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        input_min_value    = -100,
                        input_max_value    = 400,
                        target_min_value   = -300,
                        target_max_value   = 300,
                        standardize_target = False,
                        ),
                    MakeLabelOnehot(
                        channel_location = "first",
                        num_class        = num_class 
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 4
                        )
                    ]),
                "test" : Compose([
                    MinMaxStandardize(
                        input_min_value    = -100,
                        input_max_value    = 400,
                        target_min_value   = -300,
                        target_max_value   = 300,
                        standardize_target = False,
                        ),
                    AdjustDimensionality(
                        input_ndim  = 5,
                        target_ndim = 3
                        )
                    ])
 
                }

    def __call__(self, phase, image_list, label):

        return self.transforms[phase](image_list, label)


