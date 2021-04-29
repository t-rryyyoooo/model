import torchvision
from pathlib import Path

def separateData(dataset_path, criteria, phase, input_name="input*", target_name="target*"): 
    """ Extract data path based on criteria.

    Paramters: 
        dataset_path (str) -- the directory path. This directory structure must be below because we search data based on patient_id (case_00) and file name (input_, target_).

            - dataset_path
                - case_00
                    - input_0000.npy
                    - target_0000.npy
                    
                - case_01

        criteria  (dict)   -- The dict which determines which patient is assigned to train or val. Example is below.

            {
            "train" : ["00", "01", "02"],
            "val"   : ["03", "04", "05"]
            }

        phase (str)        -- determines which phase of patients to extract. train/val (depends on criteria's key)

        input_name (str)   -- for searching input image. It must be put * at the end. [ex] "input*"
        target_name (str)  -- for searching target image. It must be put *
        at the end [ex] "target*"

    Returns: 
        paired dataset
            [
            (input_0000.npy, target_0000.npy),
            (input_0001.npy, target_0001.npy)
            ]
    """
    dataset = []
    for number in criteria[phase]:
        data_path = Path(dataset_path) / ("case_" + number) 

        input_list  = data_path.glob(input_name)
        target_list = data_path.glob(target_name)
        
        input_list  = sorted(input_list)
        target_list = sorted(target_list)

        for input_file, target_file in zip(input_list, target_list):
            dataset.append((str(input_file), str(target_file)))

    return dataset


def defineModel(name):
    """ Create and return a generator.

    Parameters:
        name (str) -- Model name. [resnet50]
    """
    if name == "resnet50":
       model = torchvision.models.resnet50(progress=True) 

    else:
        raise NotImplementedError("'{}' is not supported.".format(name))

def recall(pred, true):
    tp_fn = true.sum()
    tp    = ((true == pred) * true).sum()

    score = tp / tp_fn

    return score
