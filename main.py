import torch.multiprocessing as mp
import torchvision.transforms as transforms
from foolbox.attacks import LinfPGD
from torchvision.datasets import CIFAR10

import utils
from adversarial_density import image_defence_density
from attacks import FoolboxImageAttack
from augmentations import get_augmentation
from datasets import InverseNormalize, get_torchvision_dataset_sample, BatchNormalize, get_torchvision_dataset, \
    get_index_to_class
from defence import DefendedNetwork
from models import get_model


def run_defence_experiment(dataset_class=CIFAR10, output_shape=(10, ), no_samples=50,
                           attack_class=LinfPGD, attack_params=None,
                           aug=None, sample_rate=20,
                           show=True):
    transform = get_torchvision_dataset(dataset_class, train=False).transform.transforms[2]
    index_to_class = get_index_to_class(dataset_class=dataset_class)
    model = get_model(dataset_class)

    X, y = get_torchvision_dataset_sample(dataset_class, batch_size=no_samples)
    pred = model(X)

    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)
    preprocessing = utils.normalize_to_dict(transform)
    if aug is None:
        aug = get_augmentation(dataset_class)
    image_X = inverse_transform(X)
    auged_X = aug(image_X)
    if show:
        utils.show_grid(
            [image_X[0], auged_X[0]], title="Augmentation", captions=["Image", "Augmented Image"]
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

    attack = FoolboxImageAttack(foolbox_attack_class=attack_class, params=attack_params)
    advs = attack(
        model=model, input_batch=X, true_labels=y, preprocessing=preprocessing
    )
    adv_img = advs
    advs = batch_transform(advs)
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

    utils.show_adversary_vs_original_with_preds(adv_img, img_X=image_X, y=y, adv_pred=adv_pred, pred=pred,
                                                defended_pred=defended_pred, index_to_class=index_to_class, n_show=5)

    """
    density = image_defence_density(
        model, advs, y, defence=aug, n_samples=1000, n_workers=1
    )
    robust_density = image_defence_density(
            model, advs, y, defence=aug, n_samples=1000, n_workers=1, robustness=True
        )

    de_adversarial_density = image_defence_density(
            model, advs, y, defence=aug, n_samples=1000, n_workers=1, de_adversarialize=True
        )

    print(f"Density: Avg {density.mean()}, Std: {density.std()}, Max: {density.max()}, Min: {density.min()}")
    print(f"Robust Density: Avg {robust_density.mean()}, Std: {robust_density.std()}, "
          f"Max: {robust_density.max()}, Min: {robust_density.min()}")
    print(f"De-Adversarial Density: Avg {de_adversarial_density.mean()}, Std: {de_adversarial_density.std()}, "
          f"Max: {de_adversarial_density.max()}, Min: {de_adversarial_density.min()}")
          
    """


if __name__ == "__main__":
    import torchvision.datasets as ds
    mp.set_start_method("spawn")
    device = utils.Parameters.device
    run_defence_experiment(dataset_class=ds.Caltech101, output_shape=(101,), attack_params={"epsilon": 0.00000001})