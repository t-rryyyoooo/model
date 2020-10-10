from .preprocessing import *

class UNetTransform():
    def __init__(self):
        self.transforms = {
                "train" : Compose([
                    ReadImage(), 
                    GetArrayFromImage()
                    ]), 

                "val" : Compose([
                    ReadImage(), 
                    GetArrayFromImage()
                    ])
                }

    def __call__(self, phase, image, label):
        image = self.transforms[phase](image)

        return image, label


