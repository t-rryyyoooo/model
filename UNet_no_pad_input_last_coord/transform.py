from .preprocessing import Compose, LoadMultipleData 

class UNetTransform():
    def __init__(self):

        self.transforms = {
                "train" : Compose([
                    LoadMultipleData()
                    ]), 

                "val" : Compose([
                    LoadMultipleData()
                    ])
                }

    def __call__(self, phase, image_list, label):

        return self.transforms[phase](image_list, label)


