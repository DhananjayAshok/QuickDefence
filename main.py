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


def run_defence_experiment_1(dataset_class=CIFAR10, output_shape=(10, ), no_samples=32,
                           attack_class=PGDL2, attack_params=None,
                           aug=None, sample_rate=10, density_n_samples=100,
                           show=True, density=False, indent="\t"*0):
    transform = get_torchvision_dataset(dataset_class, train=False).transform.transforms[2]
    index_to_class = get_index_to_class(dataset_class=dataset_class)
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
    defended_pred = defended.predict(X)
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
    defended_adv_pred = defended(advs).argmax(-1)
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
    return model, X, y, advs, success, success_2, batch_transform, inverse_transform


def run_defence_experiment_2(dataset_class=CIFAR10, no_samples=32,
                           attack_class=PGDL2, attack_params=None,
                           aug=None, density_n_samples=100,
                           show=True, indent="\t"*0):
    transform = get_torchvision_dataset(dataset_class, train=False).transform.transforms[2]
    model = get_model(dataset_class)

    X, y = get_torchvision_dataset_sample(dataset_class, batch_size=no_samples)
    pred = model(X).argmax(-1)

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

    attack = ImageAttack(attack_class=attack_class, params=attack_params)
    advs = attack(
        model=model, input_batch=X, true_labels=y, preprocessing=preprocessing
    )
    adv_pred = model(advs).argmax(-1)
    (
        accuracy,
        robust_accuracy,
        conditional_robust_accuracy,
        robustness,
        success,
    ) = utils.get_attack_success_measures(model, inps=X, advs=advs, true_labels=y)
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

        correct_me, correct_ma, correct_mi = utils.analyze_density_result_array(density, preprint="Correct", indent=indent)
        robust_me, robust_ma, robust_mi = utils.analyze_density_result_array(robust_density, preprint="Robust Density", indent=indent)
        de_adv_me, de_adv_ma, de_adv_mi = utils.analyze_density_result_array(de_adversarial_density, preprint="De-Adversarial Density", indent=indent)
    return utils.get_accuracy(y, pred), utils.get_accuracy(y, adv_pred), correct_me.mean(), robust_me.mean(), de_adv_me.mean()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    device = utils.Parameters.device
    import pandas as pd

    columns = ["Augmentation", "Severity", "Metric", "Value"]
    data = []

    augClass_list = [kornia.augmentation.RandomGaussianNoise, kornia.augmentation.RandomAffine]
    # Noise, ROtation and Transform
    noise_stds = [1, 2, 5]
    rotation_degrees = [5, 10, 15]
    translation_percs = [0.1, 0.2]
    dataset = ds.Caltech101
    attack_params = {"eps": 2}
    aug_list = []
    for augClass in augClass_list:
        if augClass == kornia.augmentation.RandomGaussianNoise: # If aug is noise
            for std in noise_stds:
                aug = augClass(std=std, p=1)
                aug_list.append((aug, std))
        elif augClass == kornia.augmentation.RandomAffine: # If aug is rotation
            for rotation_degree in rotation_degrees:
                aug = augClass(degrees=(-rotation_degree, rotation_degree), p=1)
                aug_list.append((aug, rotation_degree))
            for translation_perc in translation_percs:
                aug = augClass(degrees=0, translate=(0, translation_perc))
                aug_list.append((aug, translation_perc))
        else:
            pass
    for aug, sev in aug_list:
        aname = aug.__class__.__name__
        print(f"{aname}: {sev}")
        clean_accuracy, adv_accuracy, correct, robust, de_adv = \
            run_defence_experiment_2(dataset_class=dataset, no_samples=32, attack_class=PGDL2,
                                     attack_params=attack_params, density_n_samples=10, aug=aug, show=False)
        data.append([aname, sev, "Clean Accuracy", clean_accuracy])
        data.append([aname, sev, "Adv Accuracy", adv_accuracy])
        data.append([aname, sev, "Correctness", correct])
        data.append([aname, sev, "Robustness", robust])
        data.append([aname, sev, "De-Adversarialize", de_adv])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv("Experiment2.csv", index=False)




