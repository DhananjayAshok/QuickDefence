"""
We are given an adversarial example adv_inp, we want to apply a data augmentation based defence tactice defence to it
an n_sample number of times, and see how many of those are also adversarial examples for a given model.

A low defence adversary density means that often the transformed input is no longer adversarial, and this means we
would likley have a succesful defence method.
"""

import torch
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm


def image_defence_density(
    model,
    adv_image,
    true_label,
    defence=None,
    original_preds = None,
    n_samples=1000,
    n_workers=1,
    robustness=False,
    de_adversarialize = False
):
    if len(adv_image.shape) == 3:
        new_shape = (1,) + adv_image.shape
        adv_image = adv_image.reshape(new_shape)
    return defence_density(
        model,
        adv_image,
        true_label,
        defence,
        original_preds=original_preds,
        n_samples=n_samples,
        n_workers=n_workers,
        robustness=robustness,
        de_adversarialize=de_adversarialize
    )


def defence_density(
    model, adv_inps, true_labels, defence, original_preds=None, n_samples=1000, n_workers=1, robustness=False,
        de_adversarialize=False):
    """
    if robustnes is false Returns how accuracy of the model on average when defence is applied to this adv_inp
    if robustness is true returns how likely the model prediction is to change when defence is applied.

    :param model:
    :param adv_inp:
    :param true_label:
    :param defence: Augmentation to use
    :param original_pred: the predictions of the model on the non adversarial inputs
    :param n_samples:
    :param n_workers:
    :param robustness: True iff you want to measure density of defence(model)(adv_inps) = model(adv_inp)
    :param de_adversarialize: True iff you want to measure density of defence(model)(adv_inps) = original_pred
    :return:
    """
    all_results = []
    for i in tqdm(range(len(true_labels), total=len(true_labels))):
        if n_workers == 1:
            density = 0
            for n in range(n_samples):
                op = None
                if original_preds is not None:
                    op = original_preds[i]
                density += defence_density_single(
                    model, adv_inps[i:i+1], true_labels[i], defence, op,
                    robustness=robustness, de_adversarialize=de_adversarialize
                )
            all_results.append(density / n_samples)
        else:
            inp_args = (model, adv_inps[i:i+1], defence, true_labels[i], robustness)
            pool = mp.Pool(processes=n_workers)
            inps = [inp_args for i in range(n_samples)]
            res = pool.starmap(defence_density_single, inps)
            all_results.append(sum(res) / n_samples)
    all_results = np.array(all_results)
    return all_results


def defence_density_single(model, adv_inp, true_label, defence, original_pred, robustness, de_adversarialize):
    if defence is None:
        new_inp = adv_inp
    else:
        new_inp = defence(adv_inp)
    pred = model(new_inp).argmax(-1)[0]
    if robustness:
        adv_pred = model(adv_inp).argmax(-1)[0]
        succ = adv_pred != pred
    elif original_pred is not None and de_adversarialize:
        succ = pred == original_pred
    else:
        succ = pred == true_label
    if succ:
        return 1
    else:
        return 0
