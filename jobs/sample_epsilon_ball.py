"""Draw epsilon ball from adversarial example with different classes
"""
import argparse
import logging
from pathlib import Path

import init_path
import numpy as np
import torch

from attacks import ATTACKS, ImageAttack
from datasets import (
    DATASETS,
    BatchNormalize,
    get_normalization_transform,
    get_torchvision_dataset_sample,
)
from models import get_model
from utils import normalize_to_dict


def get_result_filepath():
    from init_path import parent_path

    folder = Path(parent_path) / "results" / "epsilon_ball_results"
    folder.mkdir(parents=True, exist_ok=True)

    def get_filename(idx):
        return f"{idx}.txt"

    idx = 0
    while (folder / get_filename(idx)).exists():
        idx += 1

    return folder / get_filename(idx)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("-a", "--attack", choices=ATTACKS.keys(), required=True)
    parser.add_argument("--max-epsilon", type=float, default=1e-2)
    parser.add_argument("--attack-epsilon", type=float, default=1e-2)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-num-attacks", type=int, default=100)
    parser.add_argument("--save", type=bool, default=True)
    return parser


def draw_episilon_ball(
    dataset_name,
    attack_name,
    attack_params,
    max_epsilon,
    num_samples,
    max_num_attacks,
):
    # Get model
    model = get_model(DATASETS[dataset_name])
    model.eval()

    # Get attack
    attack_class = ATTACKS[attack_name]
    attack = ImageAttack(attack_class=attack_class, params=attack_params)
    transform = get_normalization_transform(DATASETS[dataset_name])
    preprocessing = normalize_to_dict(transform)

    # Draw an image sample until adv is sucessful
    for i in range(max_num_attacks):
        X, y = get_torchvision_dataset_sample(
            DATASETS[dataset_name], train=False, batch_size=1
        )
        org_pred = model(X).argmax(-1).item()

        # Get adversarial image
        adv_X = attack(model, X, true_labels=y, preprocessing=preprocessing)
        batch_transform = BatchNormalize(normalize_transform=transform)
        adv_X = batch_transform(adv_X)
        adv_pred = model(adv_X).argmax(-1).item()

        if org_pred != adv_pred:
            break
    logging.info(f"Original prediction: {org_pred}")
    logging.info(f"Adversarial prediction: {adv_pred}")

    # Get model predictions around eps ball of adv image
    predictions = np.zeros(num_samples)
    for i in range(num_samples):
        eps = torch.rand(size=X.shape) * max_epsilon
        adv_neighbor = torch.clone(adv_X)
        adv_neighbor = adv_neighbor + eps
        predictions[i] = model(adv_neighbor).argmax(-1)

    # Collect predictions statistics
    percent_recovered = np.mean(predictions == org_pred)
    percent_adv = np.mean(predictions == adv_pred)
    percent_other = np.mean(
        np.logical_and(predictions != org_pred, predictions != adv_pred)
    )

    logging.info(
        f"Percentage of augmented attack no longer adversarial: {percent_recovered}"
    )
    logging.info(
        f"Percentage of augmented attack that remains adversarial: {percent_adv}"
    )
    logging.info(
        f"Percentage of augmented attack that remains incorrect: {percent_other}"
    )


if __name__ == "__main__":
    args = get_parser().parse_args()
    attack_params = {"eps": args.attack_epsilon}

    if args.save:
        result_filepath = get_result_filepath()
        print(f"Writing results to {result_filepath}")
        logging.basicConfig(
            filename=result_filepath, format="%(asctime)s %(message)s", level=logging.INFO
        )

    logging.info(f"Parameters: {vars(args)}")

    draw_episilon_ball(
        dataset_name=args.dataset,
        attack_name=args.attack,
        attack_params=attack_params,
        max_epsilon=args.max_epsilon,
        num_samples=args.num_samples,
        max_num_attacks=args.max_num_attacks,
    )
