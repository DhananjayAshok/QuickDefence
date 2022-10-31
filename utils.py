import matplotlib.pyplot as plt
import numpy as np
import torch
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
        assert type(img) == torch.Tensor
        img = img.cpu().detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if captions is not None:
            axs[0, i].set_xlabel(captions[i])
    if title is not None:
        plt.title(title)
    plt.show()


def get_attack_success_measures(model, inps, advs, true_labels):
    """

    :param images: list or batch of inps that can just be thrown into model
    :param advs: list or batch of adversarial inps that corresponds one to one to inps list that can be thrown in
    :param true_labels: list of integers with correct class one for each inp/advs
    :return: accuracy,
    robust_accuracy (accuracy on adversaries),
    conditional_robust_accuracy (accuracy on adversaries whos parent image is correctly classified),
    robustness (percentage of items for which the model prediction is same for inp and adv)
    success (mask vector with ith entry true iff prediction of advs[i] != prediction of inps[i]
    """
    success = []
    robust_accuracy = 0
    conditional_robust_accuracy = 0
    robustness = 0
    inp_shape = (1, ) + inps[0].shape
    n_points = len(inps)
    n_correct = 0
    for i in range(n_points):
        inp = inps[i].reshape(inp_shape)
        adv = advs[i].reshape(inp_shape)
        inp_pred = model(inp).argmax(-1)[0]
        adv_pred = model(adv).argmax(-1)[0]
        label = true_labels[i]
        correct = inp_pred == label
        pred_same = inp_pred == adv_pred
        n_correct += int(correct)
        robust_accuracy += int(correct)
        if correct:
            conditional_robust_accuracy += int(pred_same)
        robustness += int(pred_same)
        success.append(not pred_same)

    robust_accuracy = robust_accuracy/n_points
    accuracy = n_correct/n_points
    if n_correct != 0:
        conditional_robust_accuracy = conditional_robust_accuracy / n_correct
    else:
        conditional_robust_accuracy = -1
    robustness = robustness/n_points
    return accuracy, robust_accuracy, conditional_robust_accuracy, robustness, success


class Parameters:
    device = 'gpu'

