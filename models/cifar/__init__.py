import os
import torch
import utils


def get_model():
    curr_path = os.path.realpath(os.path.dirname(__file__))
    model_path = os.path.join(curr_path, "vgg_chkpnt.pth")
    model = torch.load(model_path, map_location=utils.Parameters.device)
    return model
