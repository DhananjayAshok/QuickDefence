import pandas as pd
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchattacks import PGD, PGDL2

import augmentations
import utils
from adversarial_density import image_defence_density
from attacks import ImageAttack
from augmentations import get_augmentation, kornia
from datasets import InverseNormalize, get_torchvision_dataset_sample, BatchNormalize, get_torchvision_dataset, \
    get_index_to_class
from defence import DefendedNetwork
from models import get_model
import torchvision.datasets as ds


def run_defence_experiment_1(model=None, dataset_class=CIFAR10, output_shape=(10, ), no_samples=32,
                           attack_class=PGDL2, attack_params=None,
                           aug=None, sample_rate=10, density_n_samples=100,
                           show=True, density=False, indent="\t"*0):
    transform = get_torchvision_dataset(dataset_class, train=False).transform.transforms[2]
    index_to_class = get_index_to_class(dataset_class=dataset_class)
    if model is None:
        model = get_model(dataset_class)

    X, y = get_torchvision_dataset_sample(dataset_class, batch_size=no_samples)
    pred = model(X).argmax(-1)

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
    defended_pred = defended(X)
    top = defended_pred.topk(k=2, dim=-1)[0]
    confidence = top[:, 0] - top[:, 1]
    defended_pred = defended_pred.argmax(-1)
    print(
        f"{indent}Clean Accuracy {utils.get_accuracy(y, pred)} vs Defended Accuracy "
        f"{utils.get_accuracy(y, defended_pred)}"
    )

    attack = ImageAttack(attack_class=attack_class, params=attack_params)
    advs = attack(
        model=model, input_batch=X, true_labels=y, preprocessing=preprocessing
    )
    adv_img = inverse_transform(advs)
    adv_pred = model(advs).argmax(-1)
    defended_adv_pred = defended(advs)
    top = defended_adv_pred.topk(k=2, dim=-1)[0]
    adv_confidence = top[:, 0] - top[:, 1]
    defended_adv_pred = defended_adv_pred.argmax(-1)
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
        success_2,
    ) = utils.get_attack_success_measures(defended, inps=X, advs=advs, true_labels=y)
    print(accuracy, robust_accuracy, conditional_robust_accuracy, robustness)

    if show:
        utils.show_adversary_vs_original_with_preds(adv_img, img_X=image_X, y=y, adv_pred=adv_pred, pred=pred,
                                                defended_pred=defended_adv_pred, index_to_class=index_to_class,
                                                n_show=5)
    if density:
        print(f"{indent}DENSITY EXPERIMENT RESULTS:")
        s_advs = advs[success]
        s_y = y[success]
        if sum(success) > 0:
            density = image_defence_density(
                model, s_advs, s_y, defence=aug, n_samples=density_n_samples, n_workers=1
            )
            robust_density = image_defence_density(
                    model, s_advs, s_y, defence=aug, n_samples=density_n_samples, n_workers=1, robustness=True
                )

            de_adversarial_density = image_defence_density(
                    model, s_advs, s_y, defence=aug, n_samples=density_n_samples, n_workers=1, de_adversarialize=True
                )

            utils.analyze_density_result_array(density, preprint="Density", indent=indent)
            utils.analyze_density_result_array(robust_density, preprint="Robust Density", indent=indent)
            utils.analyze_density_result_array(de_adversarial_density, preprint="De-Adversarial Density", indent=indent)
    return utils.get_accuracy(y, defended_pred), utils.get_accuracy(y, defended_adv_pred), confidence, adv_confidence


def run_defence_experiment_2(dataset_class=CIFAR10, no_samples=32,
                           attack_class=PGDL2, attack_params=None,
                           aug=None, density_n_samples=100,
                           show=True, indent="\t"*0):
    transform = get_torchvision_dataset(dataset_class, train=False).transform.transforms[2]
    model = get_model(dataset_class)

    X, y = get_torchvision_dataset_sample(dataset_class, batch_size=no_samples)

    inverse_transform = InverseNormalize(normalize_transform=transform)
    batch_transform = BatchNormalize(normalize_transform=transform)
    preprocessing = utils.normalize_to_dict(transform)
    if aug is None:
        aug = get_augmentation(dataset_class)
    image_X = inverse_transform(X)
    auged_X = batch_transform(aug(image_X))
    aug_pred = model(auged_X).argmax(-1)
    attack = ImageAttack(attack_class=attack_class, params=attack_params)
    advs = attack(
        model=model, input_batch=X, true_labels=y, preprocessing=preprocessing
    )
    adv_pred = model(advs).argmax(-1)
    aug_advs = batch_transform(aug(inverse_transform(advs)))
    aug_adv_pred = model(aug_advs).argmax(-1)

    return utils.get_accuracy(y, aug_pred), utils.get_accuracy(y, aug_adv_pred), utils.get_accuracy(adv_pred, aug_adv_pred)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    device = utils.Parameters.device
    dataset = ds.CIFAR10
    aug = augmentations.ExactL2Noise(eps=5)
    columns = ["Model", "Attack Intensity", "Accuracy", "Confidence"]
    data = []
    models = ["UU", "UA", "UN" "AU", "AA", "AN"]
    attack_intensities = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8]
    for model_t in models:
        for eps in attack_intensities:
            if model_t in ["UU", "UA", "UN"]:
                model = get_model(CIFAR10, aug=False)
            else:
                model = get_model(CIFAR10, aug=True)
            if model_t in ["UU", "AU"]:
                aug = lambda x: x
            elif model_t in ["UA", "AA"]:
                aug = get_augmentation(CIFAR10)
            else:
                aug = augmentations.CIFARAugmentation.noise
            if eps == 0:
                attack_params = {"eps": 0.1}
                clean_accuracy, adv_accuracy, confidence, adv_confidence = run_defence_experiment_1(model=model,
                                                                                         attack_params=attack_params,
                                                                                         dataset_class=dataset, aug=aug,
                                                                                         show=False)
                data.append([model_t, eps, clean_accuracy, confidence])
            else:
                attack_params = {"eps": eps}
                clean_accuracy, adv_accuracy, confidence, adv_confidence = run_defence_experiment_1(model=model,
                                                                                         attack_params=attack_params,
                                                                                         dataset_class=dataset, aug=aug,
                                                                                         show=False)
                data.append([model_t, eps, adv_accuracy, adv_confidence])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv("NewExperiment1.csv")