from .preprocessing import *

class UNetTransform():
    def __init__(self, num_class=14):
        self.transforms = {
                "train" : Compose([
                    LoadMultipleData(),
                    #MakeLabelOnehot(
                    #    channel_location = "first",
                    #    num_class        = num_class 
                    #    ),
                    AdjustDimensionality(
                        input_ndim  = 3,
                        target_ndim = 3
                        )
                    ]), 

                "val" : Compose([
                    LoadMultipleData(),
                    #MakeLabelOnehot(
                    #    channel_location = "first",
                    #    num_class        = num_class 
                    #    ),
                    AdjustDimensionality(
                        input_ndim  = 3,
                        target_ndim = 3
                        )
                    ]),
                "test" : Compose([
                    AdjustDimensionality(
                        input_ndim  = 4,
                        target_ndim = 3
                        )
                    ])
                }

    def __call__(self, phase, image, label):

        return self.transforms[phase](image, label)


