import argparse
import json
from pathlib import Path

import init_path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import ATTACKS, ImageAttack
from augmentations import get_augmentation
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
    parser.add_argument("-a", "--attack", choices=ATTACKS.keys(), required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--defence-sample-rate", type=int, default=10)
    return parser


def get_defence_results(
    dataset_name,
    attack_name,
    attack_params,
    batch_size,
    defence_sample_rate,
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
    augmentation = get_augmentation(dataset_class)

    # Get transforms
    transform = dataset.transform.transforms[2]
    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)

    all_y = torch.zeros(0)
    all_undef_pred = torch.zeros(0)
    all_undef_adv_pred = torch.zeros(0)
    all_def_pred = torch.zeros(0)
    all_def_adv_pred = torch.zeros(0)
    for (X, y) in tqdm(dataloader, total=len(dataset)):
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

        # Get defended network
        defended_model = DefendedNetwork(
            network=model,
            data_augmentation=augmentation,
            data_n_dims=3,
            transform=batch_transform,
            inverse_transform=inverse_transform,
            output_shape=(NUM_LABELS[dataset_name],),
            sample_rate=defence_sample_rate,
        )

        # Get defended predictions
        def_pred = defended_model(X).argmax(-1)
        all_def_pred = torch.cat((all_def_pred, def_pred))

        def_adv_pred = defended_model(adv_X).argmax(-1)
        all_def_adv_pred = torch.cat((all_def_adv_pred, def_adv_pred))
        break
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


if __name__ == "__main__":
    args = get_parser().parse_args()
    attack_params = {"eps": args.epsilon}
    print(f"Experiment parameters: {args}")
    print(f"{vars(args)}")

    metrics = get_defence_results(
        dataset_name=args.dataset,
        attack_name=args.attack,
        attack_params=attack_params,
        batch_size=args.batch_size,
        defence_sample_rate=args.defence_sample_rate,
    )
    print(f"Metrics:\n{metrics}")

    # Save results
    results = metrics
    results.update(vars(args))
    result_folder = Path(init_path.parent_path + f"/results/{args.attack}/{args.dataset}")
    result_folder.mkdir(parents=True, exist_ok=True)
    with open(result_folder / "result.json", "w") as f:
        json.dump(metrics, f, indent=2)
