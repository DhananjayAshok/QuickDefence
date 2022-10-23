"""
We are given an adversarial example adv_inp, we want to apply a data augmentation based defence tactice defence to it
an n_sample number of times, and see how many of those are also adversarial examples for a given model.

A low defence adversary density means that often the transformed input is no longer adversarial, and this means we
would likley have a succesful defence method.
"""

import torch


def image_defence_density(model, adv_image, true_label, defence, n_samples=1000):
    density = 0
    if len(adv_image.shape) == 3:
        new_shape = (1, ) + adv_image.shape
        adv_image = adv_image.reshape(new_shape)
    for n in range(n_samples):
        density += defence_density(model, adv_image, defence, true_label) / n_samples
    return density


def defence_density(model, adv_inp, defence, true_label):
    new_inp = defence(adv_inp)
    succ = model(new_inp).argmax(-1)[0] != true_label
    if succ:
        return 1
    else:
        return 0