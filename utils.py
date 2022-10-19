import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from eagerpy import PyTorchTensor
import os


def safe_mkdir(path, force_clean=False):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        if force_clean:
            os.rmdir(path)
            os.mkdir(path)
    return


def show_img(tensor_image, title=None):
    if type(tensor_image) == PyTorchTensor:
        tensor_image = tensor_image.raw
    plt.imshow(tensor_image.cpu().permute(1, 2, 0))
    if title is not None:
        plt.title(title)
    plt.show()


def show_grid(imgs, title=None, captions=None):
    """
    Plots a grid of all the provided images. Useful to show original and adversaries side by side.

    :param imgs: either a single image or a list of images of PyTorchTensors or Tensors (pytorch)
    :param title: string title
    :param captions: optional list of strings, must be same length as imgs
    :return: None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if type(img) == PyTorchTensor:
            img = img.raw
        img = img.cpu().detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if captions is not None:
            axs[0, i].set_xlabel(captions[i])
    if title is not None:
        plt.title(title)
    plt.show()


def get_adv_success(adv_raws, success, *args):
    """

    :param adv_raws:
    :param success:
    :param args:
    :return:
    """
    if isinstance(adv_raws, list):
        successes = []
        items = [[] for _ in args]
        for i, adv_raw in enumerate(adv_raws):
            successes.append(adv_raw[success[i]])
            for j, item in enumerate(items):
                if isinstance(args[j], list):
                    item.append(args[j][i][success[i]])
                else:
                    item.append(args[j][success[i]])
        to_ret = [successes] + items
        return to_ret
    else:
        a = [adv_raws[success]]
        return a + [item[success] for item in args]


class Parameters:
    device = 'gpu'

