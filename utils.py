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


def logit_or_pred_to_pred(pred):
    if pred is None:
        return pred
    if len(pred.shape) == 2:
        if pred.shape[1] == 1:
            return pred.view((len(pred),))
        else:
            return pred.argmax(-1)
    else:
        return pred


def show_adversary_vs_original_with_preds(
    advs, img_X, y, adv_pred, pred, defended_pred=None, n_show=5, index_to_class=None
):
    adv_pred = logit_or_pred_to_pred(adv_pred)
    pred = logit_or_pred_to_pred(pred)
    defended_pred = logit_or_pred_to_pred(defended_pred)
    if n_show <= 0:
        n_show = len(y)
    if index_to_class is None:
        index_to_class = lambda x: x
    imgs = []
    captions = []
    for i in range(n_show):
        imgs.append([advs[i], img_X[i]])
        if defended_pred is not None:
            caption = [
                f"pred={index_to_class(adv_pred[i].item())}, "
                f"defended pred={index_to_class(defended_pred[i].item())}",
                f"pred={index_to_class(pred[i].item())}, "
                f"label={index_to_class(y[i].item())}",
            ]
        else:
            caption = [
                f"pred={index_to_class(adv_pred[i].item())}, ",
                f"pred={index_to_class(pred[i].item())}, "
                f"label={index_to_class(y[i].item())}",
            ]
        # print(caption)
        captions.append(caption)
    # print(captions)
    # print(len(captions))
    # print([len(x) for x in captions])
    # print(len(imgs))
    # print([len(x) for x in imgs])

    show_grid(imgs, title="Adversarial Image vs Original Image", captions=captions)


def get_accuracy_logits(y, logits):
    return get_accuracy(y, logits.argmax(-1))


def get_accuracy(y, pred):
    return (y == pred).float().mean()


def get_accuracy_m_std(y, pred):
    return (y == pred).float().mean().item(), (y == pred).float().std().item()


def get_confidence_m_std(pred):
    top = pred.topk(k=2, dim=-1)[0]
    confidence = (top[:, 0] - top[:, 1])
    return confidence.mean().item(), confidence.std().item()


def get_conditional_robustness(y, clean_pred, adv_pred):
    """
    :return:
        conditional_robustness
        (accuracy on adversaries whos parent image is correctly classified).
    """
    y, clean_pred, adv_pred = (
        y.detach().numpy(),
        clean_pred.detach().numpy(),
        adv_pred.detach().numpy(),
    )
    idc = np.where(y == clean_pred)[0]
    return np.mean(clean_pred[idc] == adv_pred[idc])


def get_robustness(pred, adv_pred):
    pred, adv_pred = pred.detach().numpy(), adv_pred.detach().numpy()
    return np.mean(pred == adv_pred)


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
    n_correct = 0
    inp_preds = model(inps).argmax(-1)
    adv_preds = model(advs).argmax(-1)
    n_points = len(true_labels)
    for i in range(n_points):
        inp_pred = inp_preds[i]
        adv_pred = adv_preds[i]
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


def get_attack_class(attack_name="pgd"):
    if attack_name == "pgd" or attack_name == "linf":
        from torchattacks import PGD

        return PGD
    if attack_name == "pgd_l2":
        from torchattacks import PGDL2

        return PGDL2
    else:
        raise ValueError(f"Attack {attack_name} is not supported")


def measure_attack_stats(X, advs, disp=False):
    diff = (X - advs).view(X.shape[0], -1)
    l1_norm = diff.abs().sum(dim=1).mean()
    l2_norm = diff.norm(dim=1).mean()
    linf_norm = diff.abs().max(dim=1)[0].mean()
    if disp:
        print(f"L1 Norm: {l1_norm}, L2 Norm: {l2_norm}, LInf Norm: {linf_norm}")
    return l1_norm, l2_norm, linf_norm


def analyze_density_result_array(arr, preprint="", indent="", show_full=False):
    """

    :param arr: (bs, n_samples) where arr[b, i] = 1 iff iff that robustnessmmetric was 1 on the ith augmentation of X[b]
    :param preprint:
    :param indent:
    :param show_full:
    :return:
    """
    print(f"{'X'*20}")
    print(f"{indent}{preprint}")
    print(f"{'-'*20}")
    m = arr.mean(axis=1)
    ma = arr.max(axis=1)
    mi = arr.min(axis=1)
    print(f"{indent}Average Across Batches:")
    print(f"{indent}Mean: {m.mean()}, Max: {ma.mean()}, Min: {mi.mean()}")
    print(f"{indent}Maximum Across Batches:")
    print(f"{indent}Mean: {m.max()}, Max: {ma.max()}, Min: {mi.max()}")
    print(f"{indent}Minimum Across Batches:")
    print(f"{indent}Mean: {m.min()}, Max: {ma.min()}, Min: {mi.min()}")
    if show_full:
        print(arr)
    return m, ma, mi


class Parameters:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
