import __init__  # Allow executation of this file as script from parent folder
from foolbox.attacks import LinfPGD

from attacks import FoolboxImageAttack
from datasets import (
    InverseNormalize,
    get_index_to_class,
    get_normalization_transform,
    get_torchvision_dataset_sample,
)
from models import get_model
from utils import get_dataset_class, normalize_to_dict, show_grid


def test_attack_visual(
    attack_class=LinfPGD, dataset_name="mnist", n_samples=5, do_pred=False
):
    # Get model and attack
    dataset_class = get_dataset_class(dataset_name)
    model = get_model(dataset_class)
    model.eval()
    attack = FoolboxImageAttack(foolbox_attack_class=attack_class)

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


if __name__ == "__main__":
    test_attack_visual(dataset_name="cifar10", n_samples=1, do_pred=True)
