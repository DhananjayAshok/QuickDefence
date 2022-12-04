import argparse
import json
from pathlib import Path

import init_path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import ATTACKS, ImageAttack
from augmentations import get_augmentation, kornia
from datasets import (
    DATASETS,
    NUM_LABELS,
    BatchNormalize,
    InverseNormalize,
    get_torchvision_dataset,
)
from defence import DefendedNetwork
from models import get_model
from utils import (
    get_accuracy,
    get_conditional_robustness,
    get_robustness,
    normalize_to_dict,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("-a", "--attack", choices=ATTACKS.keys(), default="pdg_l2")
    parser.add_argument("-r", "--rotation-degree", type=int, default=0.0)
    parser.add_argument("-t", "--translation", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--defence-sample-rate", type=int, default=10)
    parser.add_argument("--test-defence", type=bool, default=False)
    return parser


def test_defence(
    dataset_name,
    batch_size,
    defence_sample_rate,
    rotation_degree,
    translation,
    num_batches,
):
    """Test whether defended model loses accuracy"""
    # Get dataset
    dataset_class = DATASETS[dataset_name]
    dataset = get_torchvision_dataset(dataset_class, train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get model and defence's augmentation
    model = get_model(dataset_class)
    model.eval()
    degrees = (rotation_degree, rotation_degree)
    translate = (translation, translation)
    augmentation = kornia.augmentation.RandomAffine(
        degrees=degrees, translate=translate, p=1.0
    )

    # Get defended network
    transform = dataset.transform.transforms[2]
    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)
    defended_model = DefendedNetwork(
        network=model,
        data_augmentation=augmentation,
        data_n_dims=3,
        transform=batch_transform,
        inverse_transform=inverse_transform,
        output_shape=(NUM_LABELS[dataset_name],),
        sample_rate=defence_sample_rate,
    )

    all_y = torch.zeros(0)
    all_def_pred = torch.zeros(0)

    # Get results
    num_batches_computed = 0
    for (X, y) in tqdm(dataloader, total=min(num_batches, len(dataset))):
        all_y = torch.cat((all_y, y))

        # Get defended predictions
        def_pred = defended_model(X).argmax(-1)
        all_def_pred = torch.cat((all_def_pred, def_pred))
        num_batches_computed += 1

        if num_batches_computed >= num_batches:
            break
    print(f"Defended Clean Accuracy: {get_accuracy(all_y, all_def_pred).item()}")


def get_defence_results(
    dataset_name,
    attack_name,
    attack_params,
    batch_size,
    defence_sample_rate,
    rotation_degree,
    translation,
):
    # Get dataset
    dataset_class = DATASETS[dataset_name]
    dataset = get_torchvision_dataset(dataset_class, train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Get attack
    attack_class = ATTACKS[attack_name]
    attack = ImageAttack(attack_class=attack_class, params=attack_params)

    # Get model and defence's augmentation
    model = get_model(dataset_class)
    model.eval()
    degrees = (-rotation_degree, rotation_degree)
    translate = (0, translation)
    augmentation = kornia.augmentation.RandomAffine(
        degrees=degrees, translate=translate, p=1.0
    )

    # Get defended network
    transform = dataset.transform.transforms[2]
    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)
    defended_model = DefendedNetwork(
        network=model,
        data_augmentation=augmentation,
        data_n_dims=3,
        transform=batch_transform,
        inverse_transform=inverse_transform,
        output_shape=(NUM_LABELS[dataset_name],),
        sample_rate=defence_sample_rate,
    )

    all_y = torch.zeros(0)
    all_undef_pred = torch.zeros(0)
    all_undef_adv_pred = torch.zeros(0)
    all_def_pred = torch.zeros(0)
    all_def_adv_pred = torch.zeros(0)
    for (X, y) in tqdm(dataloader, total=len(dataloader)):
        all_y = torch.cat((all_y, y))

        # Get undefended clean predictions
        undef_pred = model(X).argmax(-1)
        all_undef_pred = torch.cat((all_undef_pred, undef_pred))

        # Get adversarial image
        preprocessing = normalize_to_dict(transform)
        adv_X = attack(
            model=model, input_batch=X, true_labels=y, preprocessing=preprocessing
        )
        adv_X = batch_transform(adv_X)

        # Get undefended adv predictions
        undef_adv_pred = model(adv_X).argmax(-1)
        all_undef_adv_pred = torch.cat((all_undef_adv_pred, undef_adv_pred))

        # Get defended predictions
        def_pred = defended_model(X).argmax(-1)
        all_def_pred = torch.cat((all_def_pred, def_pred))

        def_adv_pred = defended_model(adv_X).argmax(-1)
        all_def_adv_pred = torch.cat((all_def_adv_pred, def_adv_pred))

    metrics = {
        "Undefended Clean Accuracy": get_accuracy(all_y, all_undef_pred).item(),
        "Undefended Advsarial Accuracy": get_accuracy(all_y, all_undef_adv_pred).item(),
        "Undefended Advsarial Robustness": get_robustness(
            all_undef_pred,
            all_undef_adv_pred,
        ).item(),
        "Undefended Advsarial Conditional Robustness": get_conditional_robustness(
            all_y,
            all_undef_pred,
            all_undef_adv_pred,
        ).item(),
        "Defended Clean Accuracy": get_accuracy(all_y, all_def_pred).item(),
        "Defended Advsarial Accuracy": get_accuracy(all_y, all_def_adv_pred).item(),
        "Defended Advsarial Robustness": get_robustness(
            all_def_pred,
            all_def_adv_pred,
        ).item(),
        "Defended Advsarial Conditional Robustness": get_conditional_robustness(
            all_y,
            all_def_pred,
            all_def_adv_pred,
        ).item(),
    }

    return metrics


def get_result_filepath(attack_name, dataset_name):
    from init_path import parent_path

    folder = (
        Path(parent_path) / "results" / "defence_results" / attack_name / dataset_name
    )
    folder.mkdir(parents=True, exist_ok=True)

    def get_filename(idx):
        return f"{idx}.json"

    idx = 0
    while (folder / get_filename(idx)).exists():
        idx += 1

    return folder / get_filename(idx)


if __name__ == "__main__":
    args = get_parser().parse_args()
    attack_params = {"eps": args.epsilon}
    print(f"Experiment parameters: {args}")
    print(f"{vars(args)}")

    if not args.test_defence:
        metrics = get_defence_results(
            dataset_name=args.dataset,
            attack_name=args.attack,
            attack_params=attack_params,
            batch_size=args.batch_size,
            defence_sample_rate=args.defence_sample_rate,
            rotation_degree=args.rotation_degree,
            translation=args.translation,
        )
        print(f"Metrics:\n{metrics}")

        # Save results
        results = metrics
        results.update(vars(args))
        filepath = get_result_filepath(args.attack, args.dataset)
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        test_defence(
            args.dataset,
            args.batch_size,
            args.defence_sample_rate,
            args.rotation_degree,
            args.translation,
            args.num_batches,
        )
