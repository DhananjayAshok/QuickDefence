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
    return utils.get_accuracy(y, defended_pred), utils.get_accuracy(adv_pred, defended_adv_pred), confidence, adv_confidence


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
    attack_params = {"eps": 2}
    columns = ["Model", "Adversarial", "Augmentation", "Intensity",  "Accuracy", "Confidence"]
    data = []
    models = ["U", "A"]
    augmentation_set = ["Noise", "Rotation", "Translation"]
    for model_t in models:
        if model_t in ["U"]:
            model = get_model(CIFAR10, aug=False)
        else:
            model = get_model(CIFAR10, aug=True)
        for augmentation in augmentation_set:
            if augmentation == "Noise":
                for eps in [0.01, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 25]:
                    aug = augmentations.ExactL2Noise(eps=eps)
                    defended_accuracy, adv_robustness, confidence, adv_confidence = run_defence_experiment_1(model=model,
                                                                                     attack_params=attack_params,
                                                                                     dataset_class=dataset, aug=aug,
                                                                                     show=False)
                    data.append([model_t, False, defended_accuracy, eps, defended_accuracy, confidence])
                    data.append([model_t, True, adv_robustness, augmentations, eps, adv_confidence])
            elif augmentation == "Rotation":
                for degrees in [5, 10, 15, 20, 35, 45, 55, 65, 75, 90]:
                    aug = kornia.augmentation.RandomAffine(degrees=degrees, p=1)
                    defended_accuracy, adv_robustness, confidence, adv_confidence = run_defence_experiment_1(
                        model=model,
                        attack_params=attack_params,
                        dataset_class=dataset, aug=aug,
                        show=False)
                    data.append([model_t, False, defended_accuracy, degrees, defended_accuracy, confidence])
                    data.append([model_t, True, adv_robustness, augmentations, degrees, adv_confidence])
            elif augmentation == "Translation":
                for translate in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    aug = kornia.augmentation.RandomAffine(degrees=1, translate=(translate, translate), p=1)
                    defended_accuracy, adv_robustness, confidence, adv_confidence = run_defence_experiment_1(
                        model=model,
                        attack_params=attack_params,
                        dataset_class=dataset, aug=aug,
                        show=False)
                    data.append([model_t, False, defended_accuracy, translate, defended_accuracy, confidence])
                    data.append([model_t, True, adv_robustness, augmentations, translate, adv_confidence])

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv("NewExperiment2.csv")