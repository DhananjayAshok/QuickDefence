import torch
import os
import utils


def get_model(ds_class):
    curr_path = os.path.realpath(os.path.dirname(__file__))
    model_path = os.path.join(curr_path, f"{ds_class.__name__}.pth")
    model = torch.load(model_path, map_location=utils.Parameters.device)
    return model

