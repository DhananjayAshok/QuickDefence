"""
This is taken from the foolbox system, but we might want to come up with a more different way to get attacks?
We should TODO: Decide what attacks to benchmark against and implement those. (One or two for each type max)
Current List of options:
    1. Pertubation Attack - PGD, ?
    2. Translation / Spatial Attack - Foolbox implementaiton, ?
    3. Colour Attack - ColourFool, ?

"""
import torch.nn as nn
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import SpatialAttack
from foolbox import accuracy
import utils


class FoolboxImageAttack:
    def __init__(self, foolbox_attack_class, params=None):
        self.params = params
        self.foolbox_attack_class = foolbox_attack_class
        self.spatial = foolbox_attack_class == SpatialAttack
        if params is None:
            if self.spatial:
                self.params = {'max_translation': 2, 'num_translations': 2, 'max_rotation': 10, 'num_rotations': 2}
            else:
                self.params = {'epsilon': 0.00001}

    def __call__(self, model, input_batch, true_labels, target_labels=None, preprocessing=None):
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, device=utils.Parameters.device)
        images, labels = ep.astensors(input_batch, true_labels)
        if not self.spatial:
            attack = self.foolbox_attack_class()
            epsilons = [self.params['epsilon']]
            raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
            raw_advs = [r.raw for r in raw_advs][0]
        else:
            attack = self.foolbox_attack_class(max_translation=self.params['max_translation'],
                                               num_translations=self.params['num_translations'],
                                               max_rotation=self.params['max_rotation'],
                                               num_rotations=self.params['num_rotations'])
            raw_advs, _, _ = attack(fmodel, images, labels)
            raw_advs = raw_advs.raw
        return raw_advs
