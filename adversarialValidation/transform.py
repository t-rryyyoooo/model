from .preprocessing import *

class CNNTransform():
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

    def __call__(self, phase, image):
        image = self.transforms[phase](image)

        return image


