"""Draw epsilon ball from adversarial example with different classes
"""
import argparse

import init_path
import numpy as np
import torch

from attacks import ATTACKS, FoolboxImageAttack
from datasets import (
    DATASETS,
    BatchNormalize,
    get_normalization_transform,
    get_torchvision_dataset_sample,
)
from models import get_model
from utils import normalize_to_dict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("-a", "--attack", choices=ATTACKS.keys(), required=True)
    parser.add_argument("--num-tiles", type=lambda x: int(x) + int(x) % 2, default=10)
    parser.add_argument("--noise-epsilon", type=float, default=1e-2)
    parser.add_argument("--attack-epsilon", type=float, default=1e-2)
    return parser


def draw_episilon_ball(
    dataset_name,
    attack_name,
    attack_params,
    num_tiles,
    noise_epsilon,
    pixel_i,
    pixel_j,
    color_channel,
):
    # Get model
    model = get_model(DATASETS[dataset_name])
    model.eval()

    # Draw an image sample
    X, y = get_torchvision_dataset_sample(
        DATASETS[dataset_name], train=False, batch_size=1
    )
    org_pred = model(X).argmin(-1)
    print(f"Original prediction: {org_pred}")

    # Get adversarial image
    attack_class = ATTACKS[attack_name]
    attack = FoolboxImageAttack(foolbox_attack_class=attack_class, params=attack_params)
    transform = get_normalization_transform(DATASETS[dataset_name])
    preprocessing = normalize_to_dict(transform)
    adv_X = attack(model, X, true_labels=y, preprocessing=preprocessing)
    batch_transform = BatchNormalize(normalize_transform=transform)
    adv_X = batch_transform(adv_X)
    adv_pred = model(adv_X).argmin(-1)
    print(f"Adversarial prediction: {adv_pred}")

    # Get model predictions around eps ball of adv image
    color_channel = min(color_channel, X.shape[0] - 1)
    dim = X.shape[2]
    eps = [noise_epsilon * i for i in range(-num_tiles // 2, num_tiles // 2)]
    predictions = np.zeros((num_tiles, num_tiles))

    for i in range(num_tiles):
        for j in range(num_tiles):
            adv_neighbor = torch.clone(adv_X)
            adv_neighbor[0, color_channel, pixel_i // dim, pixel_i % dim] += eps[i]
            adv_neighbor[0, color_channel, pixel_j // dim, pixel_j % dim] += eps[j]
            print(f"Adv shape {adv_neighbor.shape}")
            predictions[i, j] = model(adv_neighbor).argmin(-1)
    print(f"predictions: {predictions}")

    # Draw model predictions
    pass


if __name__ == "__main__":
    args = get_parser().parse_args()
    attack_params = {"epsilon": args.attack_epsilon}
    draw_episilon_ball(
        dataset_name=args.dataset,
        attack_name=args.attack,
        attack_params=attack_params,
        num_tiles=args.num_tiles,
        noise_epsilon=args.noise_epsilon,
        pixel_i=0,
        pixel_j=2,
        color_channel=0,
    )
