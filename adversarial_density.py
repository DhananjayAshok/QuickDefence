"""
We are given an adversarial example adv_inp, we want to apply a data augmentation based defence tactice defence to it
an n_sample number of times, and see how many of those are also adversarial examples for a given model.

A low defence adversary density means that often the transformed input is no longer adversarial, and this means we
would likley have a succesful defence method.
"""

import torch
import numpy as np
import torch.multiprocessing as mp


def image_defence_density(
    model,
    adv_image,
    true_label,
    defence=None,
    n_samples=1000,
    n_workers=1,
    robustness=False,
):
    if len(adv_image.shape) == 3:
        new_shape = (1,) + adv_image.shape
        adv_image = adv_image.reshape(new_shape)
    return defence_density(
        model,
        adv_image,
        true_label,
        defence,
        n_samples,
        n_workers=n_workers,
        robustness=robustness,
    )


def defence_density(
    model, adv_inps, true_labels, defence, n_samples=1000, n_workers=1, robustness=False
):
    """
    if robustnes is false Returns how accuracy of the model on average when defence is applied to this adv_inp
    if robustness is true returns how likely the model prediction is to change when defence is applied.

    :param model:
    :param adv_inp:
    :param true_label:
    :param defence:
    :param n_samples:
    :param n_workers:
    :param robustness:
    :return:
    """
    all_results = []
    for i in range(len(true_labels)):
        if n_workers == 1:
            density = 0
            for n in range(n_samples):
                density += defence_density_single(
                    model, adv_inps[i], true_labels[i], defence, robustness=robustness
                )
            all_results.append(density / n_samples)
        else:
            inp_args = (model, adv_inps[i], defence, true_labels[i], robustness)
            pool = mp.Pool(processes=n_workers)
            inps = [inp_args for i in range(n_samples)]
            res = pool.starmap(defence_density_single, inps)
            all_results.append(sum(res) / n_samples)
    all_results = np.array(all_results)
    return all_results


def defence_density_single(model, adv_inp, true_label, defence, robustness):
    if defence is None:
        new_inp = adv_inp
    else:
        new_inp = defence(adv_inp)
    pred = model(new_inp).argmax(-1)[0]
    if robustness:
        adv_pred = model(adv_inp).argmax(-1)[0]
        succ = adv_pred != pred
    else:
        succ = pred == true_label
    if succ:
        return 1
    else:
        return 0
