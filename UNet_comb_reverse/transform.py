if __name__ == "__main__":
    from preprocessing import *
else:
    from .preprocessing import *

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


