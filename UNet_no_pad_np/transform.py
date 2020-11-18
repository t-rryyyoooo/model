from .preprocessing import Compose, LoadNumpys

class UNetTransform():
    def __init__(self):

        self.transforms = {
                "train" : Compose([
                    LoadNumpys()
                    ]), 

                "val" : Compose([
                    LoadNumpys()
                    ])
                }

    def __call__(self, phase, image, label):

        return self.transforms[phase](image, label)


