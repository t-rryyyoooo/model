from .preprocessing import Compose, LoadMultipleData , MinMaxStandardize, AdjustDimensionality

class UNetTransform():
    def __init__(self):

        self.transforms = {
                "train" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        input_min_value  = [-300, -200],
                        input_max_value  = [300, 500],
                        target_min_value = -300,
                        target_max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 3
                        )
                    ]), 

                "val" : Compose([
                    LoadMultipleData(),
                    MinMaxStandardize(
                        input_min_value  = [-300, -200],
                        input_max_value  = [300, 500],
                        target_min_value = -300,
                        target_max_value = 300
                        ),
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 3
                        )
                    ])
                }

    def __call__(self, phase, image_list, label):

        return self.transforms[phase](image_list, label)


