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
