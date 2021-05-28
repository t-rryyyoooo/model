from .preprocessing import *

class UNetTransform():
    def __init__(self, test_mix_rate=0.0):

        self.transforms = {
                "train" : Compose([
                    LoadMultipleData(),
                    ClipValues(
                        input_min_value  = [-300, -300, None],
                        input_max_value  = [300, 300, None],
                        target_min_value = None,
                        target_max_value = None 
                        ),
                    StackImages(
                        target_numbers = [0, 1],
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 3
                        )
                    ]), 
                "val" : Compose([
                    LoadMultipleData(),
                    ClipValues(
                        input_min_value  = [-300, -300, None],
                        input_max_value  = [300, 300, None],
                        target_min_value = None,
                        target_max_value = None 
                        ),
                    StackImages(
                        target_numbers = [0, 1],
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 3
                        )
                    ]),
                "test" : Compose([
                    ClipValues(
                        input_min_value  = [-300, -300, None],
                        input_max_value  = [300, 300, None],
                        target_min_value = None,
                        target_max_value = None 
                        ),
                    MixImages(
                        target_numbers = [0, 1],
                        mode           = "static",
                        constant_value = test_mix_rate
                        ),
                    AdjustDimensionality(
                        input_ndim  = 5,
                        target_ndim = 3
                        )]) 
                }

    def __call__(self, phase, image_list, label):

        return self.transforms[phase](image_list, label)


