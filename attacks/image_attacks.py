"""
This is taken from the foolbox system, but we might want to come up with a more different way to get attacks?
We should TODO: Decide what attacks to benchmark against and implement those. (One or two for each type max)
Current List of options:
    1. Pertubation Attack - PGD, ?
    2. Translation / Spatial Attack - Foolbox implementaiton, ?
    3. Colour Attack - ColourFool, ?

"""

from attacks.attack import Attack
import torch.nn as nn
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import SpatialAttack
from foolbox import accuracy
import utils


class ImagePerturbAttack(Attack):
    def __init__(self, foolbox_attack_class):
        self.foolbox_attack_class = foolbox_attack_class
        Attack.__init__(self)

    def __call__(self, model, input_batch, true_labels, target_labels=None, epsilon=0.00001, preprocessing=None):
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, device=utils.Parameters.device)
        images, labels = ep.astensors(input_batch, true_labels)
        # apply the attack
        attack = self.foolbox_attack_class()
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=[epsilon])
        raw_advs = [r.raw for r in raw_advs][0]
        return raw_advs


class ImageSpatialAttack(Attack):
    def __init__(self, max_translation=2, num_translations=2, max_rotation=10, num_rotations=2):
        self.attack = SpatialAttack(
            max_translation=max_translation,  # 6px so x in [x-6, x+6] and y in [y-6, y+6]
            num_translations=num_translations,  # number of translations in x, y.
            max_rotation=max_rotation,  # +- rotation in degrees
            num_rotations=num_rotations,  # number of rotations
            # max total iterations = num_rotations * num_translations**2
        )

        # report the success rate of the attack (percentage of samples that could
        # be adversarially perturbed) and the robust accuracy (the remaining accuracy
        # of the model when it is attacked)

    def __call__(self, model, input_batch, true_labels, preprocessing=None):
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, device=utils.Parameters.device)
        images, labels = ep.astensors(input_batch, true_labels)
        xp_, _, success = self.attack(fmodel, images, labels)
        return xp_.raw

