from attacks.attack import Attack
import torch.nn as nn
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import SpatialAttack
from foolbox import accuracy


class ImagePerturbAttack(Attack):
    def __init__(self, foolbox_attack_class):
        self.foolbox_attack_class = foolbox_attack_class
        Attack.__init__(self)

    def __call__(self, model, input_batch, true_labels, target_labels=None, preprocessing=
    dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3), epsilons=[0.0002, 0.01]):
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        images, labels = ep.astensors(input_batch, true_labels)
        clean_accuracy = accuracy(fmodel, images, labels)
        print(f"Foolbox clean accuracy {clean_accuracy}")
        # apply the attack
        attack = self.foolbox_attack_class()
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        adv_preds = [fmodel(r).argmax(-1).raw for r in raw_advs]
        raw_advs = [r.raw for r in raw_advs]
        return raw_advs, adv_preds, success.raw, 1 - success.float32().mean(axis=-1)


class ImageSpatialAttack(Attack):
    def __init__(self, max_translation=6, num_translations=6, max_rotation=20, num_rotations=5):
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

    def __call__(self, model, input_batch, true_labels, preprocessing=
    dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)):
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        images, labels = ep.astensors(input_batch, true_labels)
        xp_, _, success = self.attack(fmodel, images, labels)
        adv_preds = fmodel(xp_).argmax(-1).raw
        suc = success.float32().mean().item() * 100
        return xp_.raw, adv_preds, success.raw, 100-suc

