import torch.multiprocessing as mp
import torchvision.transforms as transforms
from foolbox.attacks import LinfPGD
from torchvision.datasets import CIFAR10

import utils
from adversarial_density import image_defence_density
from attacks import FoolboxImageAttack
from augmentations import get_augmentation
from datasets import InverseNormalize, get_torchvision_dataset_sample, BatchNormalize, get_torchvision_dataset
from defence import DefendedNetwork
from models import get_model


def run_defence_experiment(dataset_class=CIFAR10, output_shape=(10, ), attack_class=LinfPGD, no_samples=32,
                           sample_rate=10, show=True):
    transform = get_torchvision_dataset(dataset_class, train=False).transform.transforms[2]
    model = get_model(dataset_class)

    X, y = get_torchvision_dataset_sample(dataset_class, batch_size=no_samples)
    X, y = X.to(device), y.to(device)
    pred = model(X)

    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)
    preprocessing = utils.normalize_to_dict(transform)
    aug = get_augmentation(dataset_class)
    image_X = inverse_transform(X)[0]
    auged_X = aug(image_X)[0]
    if show:
        utils.show_grid(
            [image_X, auged_X], title="Augmentation", captions=["Image", "Augmented Image"]
        )
    defended = DefendedNetwork(
        network=model,
        data_augmentation=aug,
        data_n_dims=3,
        transform=batch_transform,
        inverse_transform=inverse_transform,
        output_shape=output_shape,
        sample_rate=sample_rate,
    )
    defended_pred = defended.predict(X)
    print(
        f"Clean Accuracy {utils.get_accuracy_logits(y, pred)} vs Defended Accuracy "
        f"{utils.get_accuracy(y, defended_pred)}"
    )

    attack = FoolboxImageAttack(foolbox_attack_class=attack_class)
    advs = attack(
        model=model, input_batch=X, true_labels=y, preprocessing=preprocessing
    )
    adv_pred = model(advs).argmin(-1)
    (
        accuracy,
        robust_accuracy,
        conditional_robust_accuracy,
        robustness,
        success,
    ) = utils.get_attack_success_measures(model, inps=X, advs=advs, true_labels=y)
    print(accuracy, robust_accuracy, conditional_robust_accuracy, robustness)
    (
        accuracy,
        robust_accuracy,
        conditional_robust_accuracy,
        robustness,
        success,
    ) = utils.get_attack_success_measures(defended, inps=X, advs=advs, true_labels=y)
    print(accuracy, robust_accuracy, conditional_robust_accuracy, robustness)

    utils.show_grid(
        [advs[0], image_X],
        title="Adversarial Image vs Original Image",
        captions=[f"pred={adv_pred[0].argmax(-1).data}, defended pred={defended_pred[0].data}",
                  f"pred={pred[0].argmax(-1).data}, label={y[0].argmax(-1).data}"],
    )
    print(
        image_defence_density(
            model, advs[0], y[0], n_samples=1000, n_workers=1
        )
    )
    print(
        image_defence_density(
            model, advs[0], y[0], n_samples=1000, n_workers=1, robustness=True
        )
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    device = utils.Parameters.device
    run_defence_experiment()