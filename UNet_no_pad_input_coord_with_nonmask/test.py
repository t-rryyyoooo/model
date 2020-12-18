from utils import separateDataWithNonMask


criteria = {
        "train" : ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"],
        "val" : ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]
        }

rate = {
        "train" : {"mask" : 1.0, "nonmask" : 0.1},
        "val" : {"mask" : 0.1, "nonmask" : 0.1}
        }

dataset = separateDataWithNonMask("/sandisk/data/patch/Abdomen/no_pad/coordinate/116-132-132-1-3/liver/mask/image", "/sandisk/data/patch/Abdomen/no_pad/coordinate/116-132-132-1-3/liver/mask/image", criteria, "val", rate)
print(len(dataset))
