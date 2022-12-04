import torch
import os
import utils
import torchvision.datasets as ds


def get_model(ds_class, aug=False):
    if ds_class == ds.CIFAR10:
        suffix=f"_"
        if aug:
            suffix += "_aug"
    else:
        suffix = ""
    curr_path = os.path.realpath(os.path.dirname(__file__))
    model_path = os.path.join(curr_path, f"{ds_class.__name__}{suffix}.pth")
    model = torch.load(model_path, map_location=utils.Parameters.device)
    return model

