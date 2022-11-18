import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from eagerpy import PyTorchTensor


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

    :param imgs: either a single image or a list of images of PyTorchTensors or Tensors (pytorch) or a list of lists
    :param title: string title
    :param captions: optional list of strings, must be same shape as imgs
    :return: None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(imgs[0], list):
        imgs = [imgs]
    if isinstance(captions, list) and not isinstance(captions[0], list):
        captions = [captions]
    # By now the imgs is a nested list. A single list input gets sent to n_rows 1 and n_columns len(input_imgs)
    n_rows = len(imgs)
    n_cols = len(imgs[0])
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, squeeze=False)
    for i, img_row in enumerate(imgs):
        for j, img in enumerate(img_row):
            if type(img) == PyTorchTensor:
                img = img.raw
            assert type(img) == torch.Tensor
            img = img.cpu().detach()
            img = F.to_pil_image(img)
            axs[i, j].imshow(np.asarray(img))
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if captions is not None:
                axs[i, j].set_xlabel(str(captions[i][j]))
    if title is not None:
        fig.suptitle(title)
    plt.show()


def get_accuracy_logits(y, logits):
    return get_accuracy(y, logits.argmax(-1))


def get_accuracy(y, pred):
    return (y == pred).float().mean()


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
    inp_shape = (1,) + inps[0].shape
    n_points = len(inps)
    n_correct = 0
    for i in range(n_points):
        inp = inps[i].reshape(inp_shape)
        adv = advs[i].reshape(inp_shape)
        inp_pred = model(inp).argmax(-1)[0]
        adv_pred = model(adv).argmax(-1)[0]
        label = true_labels[i]
        correct = inp_pred == label
        adv_correct = adv_pred == label
        pred_same = inp_pred == adv_pred
        n_correct += int(correct)
        robust_accuracy += int(adv_correct)
        if correct:
            conditional_robust_accuracy += int(pred_same)
        robustness += int(pred_same)
        success.append(not pred_same)

    robust_accuracy = robust_accuracy / n_points
    accuracy = n_correct / n_points
    if n_correct != 0:
        conditional_robust_accuracy = conditional_robust_accuracy / n_correct
    else:
        conditional_robust_accuracy = -1
    robustness = robustness / n_points
    return accuracy, robust_accuracy, conditional_robust_accuracy, robustness, success


def repeat_batch_images(x, num_repeat):
    """Receives a batch of images and repeat each image for num_repeat times
    :param x: Images of shape (B, C, H, W)
    :return: Images of shape (Bxnum_repeat, C, H, W) where each image is repeated
        along the first dimension for num_repeat times
    """
    assert len(x.shape) == 4
    x = x.unsqueeze(1).repeat(1, num_repeat, 1, 1, 1)
    x = x.view((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
    return x


def normalize_to_dict(normalize):
    preprocessing = dict(
        mean=list(normalize.mean), std=list(normalize.std), axis=-(len(normalize.mean))
    )
    return preprocessing


def get_dataset_class(dataset_name="mnist"):
    if dataset_name == "mnist":
        from torchvision.datasets import MNIST

        return MNIST
    elif dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10

        return CIFAR10
    elif dataset_name == "caltech101":
        from torchvision.datasets import Caltech101

        return Caltech101
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")


class Parameters:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
