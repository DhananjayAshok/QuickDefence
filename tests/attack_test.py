import __init__  # Allow executation of this file as script from parent folder
import torch
from foolbox.attacks import LinfPGD

import utils
from attacks import FoolboxImageAttack
from datasets import (
    BatchNormalize,
    InverseNormalize,
    get_index_to_class,
    get_normalization_transform,
    get_torchvision_dataset_sample,
)
from models import get_model
from utils import get_dataset_class, normalize_to_dict, show_grid


def test_attack_visual(
    attack_class=LinfPGD,
    dataset_name="mnist",
    n_samples=5,
    do_pred=False,
    attack_params=None,
):
    # Get model and attack
    dataset_class = get_dataset_class(dataset_name)
    model = get_model(dataset_class).to(utils.Parameters.device)
    model.eval()
    attack = FoolboxImageAttack(foolbox_attack_class=attack_class, params=attack_params)

    # Get samples to apply attack
    normalize_transform = get_normalization_transform(dataset_class)
    X, y = get_torchvision_dataset_sample(dataset_class, batch_size=n_samples)
    if do_pred:
        pred = model(X).argmax(-1)
    inv_norm_X = InverseNormalize(normalize_transform=normalize_transform)(X)

    # Get adversarial examples
    preprocessing = normalize_to_dict(normalize_transform)
    adv_x = attack(model, input_batch=X, true_labels=y, preprocessing=preprocessing)
    if do_pred:
        adv_pred = model(adv_x).argmax(-1)

    # Build image grid
    grid = []
    if do_pred:
        captions = []
        index_to_class = get_index_to_class(dataset_name)
    for i in range(n_samples):
        img_row = [inv_norm_X[i, :, :, :], adv_x[i, :, :, :]]
        if do_pred:
            pred_label, adv_label = (
                index_to_class(pred[i].item()),
                index_to_class(adv_pred[i].item()),
            )
            captions.append([f"{pred_label}", f"{adv_label}"])
        grid.append(img_row)

    show_grid(
        grid,
        title="Original Images vs. Adversarial Images",
        captions=captions if do_pred else None,
    )


def test_attack_maginitude(dataset_name, attack_name, attack_params):
    # Get model and attack
    dataset_class = get_dataset_class(dataset_name)
    attack_class = utils.get_attack_class(attack_name)
    model = get_model(dataset_class).to(utils.Parameters.device)
    model.eval()
    attack = FoolboxImageAttack(foolbox_attack_class=attack_class, params=attack_params)

    # Get samples to apply attack
    normalize_transform = get_normalization_transform(dataset_class)
    inverse_transform = InverseNormalize(normalize_transform=normalize_transform)
    X, y = get_torchvision_dataset_sample(dataset_class, batch_size=1)
    image_X = inverse_transform(X)

    # Get adversarial examples
    preprocessing = normalize_to_dict(normalize_transform)
    adv_x = attack(model, input_batch=X, true_labels=y, preprocessing=preprocessing)
    # batch_transform = BatchNormalize(normalize_transform=normalize_transform)
    # adv_x = batch_transform(adv_x)

    diff = torch.abs(image_X - adv_x)
    print(f"Image dimension: {image_X.shape}")
    print(f"Max perturbation: {torch.max(diff)}")
    print(f"Perturbation L1 norm: {torch.sum(diff)}")
    print(f"Mean: {torch.mean(diff)}")


if __name__ == "__main__":
    test_attack_maginitude(
        "caltech101", attack_name="linf", attack_params={"epsilon": 1e-2}
    )
    # test_attack_visual(
    #     dataset_name="caltech101",
    #     n_samples=1,
    #     do_pred=True,
    #     attack_params={"epsilon": 0.0000001},
    # )
