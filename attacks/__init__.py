"""
This is taken from the foolbox system, but we might want to come up with a more different way to get attacks?
We should TODO: Decide what attacks to benchmark against and implement those. (One or two for each type max)
Current List of options:
    1. Pertubation Attack - PGD, ?
    2. Translation / Spatial Attack - Foolbox implementaiton, ?
    3. Colour Attack - ColourFool, ?

"""
from torchattacks import PGD, PGDL2
import utils

ATTACKS = {"pgd": PGD, "pdg_l2": PGDL2}


class ImageAttack:
    def __init__(self, attack_class, params=None):
        self.params = params
        self.attack_class = attack_class
        if params is None:
            self.params = {"eps": 0.001}

    def __call__(
        self, model, input_batch, true_labels, target_labels=None, preprocessing=None
    ):
        attack = self.attack_class(model, **self.params)
        attack.set_normalization_used(mean=list(preprocessing['mean']), std=list(preprocessing['std']))
        return attack(input_batch, true_labels)
