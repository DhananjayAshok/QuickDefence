import pandas as pd
from torchvision.datasets import CIFAR10
from torchattacks import PGD, PGDL2, TPGD

import augmentations
import utils
from attacks import ImageAttack
from augmentations import get_augmentation, kornia
from datasets import InverseNormalize, get_torchvision_dataset_sample, BatchNormalize, get_torchvision_dataset
from defence import DefendedNetwork
from models import get_model
import torchvision.datasets as ds
from tqdm import tqdm


def experiment(model=None, dataset_class=CIFAR10, output_shape=(10, ), batch_size=100, X=None, y=None,
               attack_class=PGDL2, attack_params=None, transform=None,
               aug=None, sample_rate=10):
    if model is None:
        model = get_model(dataset_class)
    if aug is None:
        aug = get_augmentation(dataset_class)
    if X is None or y is None:
        X, y = get_torchvision_dataset_sample(dataset_class, batch_size=batch_size)
    if transform is None:
        transform = get_torchvision_dataset(dataset_class, train=False).transform.transforms[2]
    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)
    preprocessing = utils.normalize_to_dict(transform)

    defended = DefendedNetwork(
        network=model,
        data_augmentation=aug,
        data_n_dims=3,
        transform=batch_transform,
        inverse_transform=inverse_transform,
        output_shape=output_shape,
        sample_rate=sample_rate,
    )

    attack = ImageAttack(attack_class=attack_class, params=attack_params)
    advs = attack(
        model=model, input_batch=X, true_labels=y, preprocessing=preprocessing
    )

    pred = model(X)
    adv_pred = model(advs)
    defended_pred = defended(X)
    defended_adv_pred = defended(advs)

    confidence = utils.get_confidence_m_std(pred)
    adv_confidence = utils.get_confidence_m_std(adv_pred)
    defended_confidence = utils.get_confidence_m_std(defended_pred)
    defended_adv_confidence = utils.get_confidence_m_std(defended_adv_pred)

    pred = pred.argmax(-1)
    adv_pred = adv_pred.argmax(-1)
    defended_pred = defended_pred.argmax(-1)
    defended_adv_pred = defended_adv_pred.argmax(-1)

    clean_accuracy = utils.get_accuracy_m_std(y, pred)
    adv_accuracy = utils.get_accuracy_m_std(y,adv_pred)
    defended_accuracy = utils.get_accuracy_m_std(y, defended_pred)
    defended_adv_accuracy = utils.get_accuracy_m_std(y, defended_adv_pred)
    defended_adv_robust = utils.get_accuracy_m_std(defended_adv_pred, adv_pred)

    return clean_accuracy, adv_accuracy, defended_accuracy, defended_adv_accuracy, \
        defended_adv_robust, confidence, adv_confidence, defended_confidence, \
        defended_adv_confidence


def run_experiment1(save_name="Experiment1", precompute_data=True, batch_size=100, n_runs=5, attack_class=PGDL2):
    print(f"Running Experiment 1 and saving to {save_name}_{attack_class.__name__}.csv")
    columns = ["Run", "Model", "Attack Intensity", "Accuracy", "Confidence"]
    data = []
    dataset = ds.CIFAR10
    transform = get_torchvision_dataset(dataset_class=dataset, train=False).transform.transforms[2]
    for i in tqdm(range(n_runs)):
        X = None
        y = None
        if precompute_data:
            X, y = get_torchvision_dataset_sample(dataset_class=dataset, batch_size=batch_size)
        models = ["UU", "UA", "UN", "AU", "AA", "AN"]
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
                attack_params = {"eps": eps}
                clean_accuracy, adv_accuracy, defended_accuracy, defended_adv_accuracy, \
                    defended_adv_robust, confidence, adv_confidence, defended_confidence, \
                    defended_adv_confidence = experiment(model=model, attack_class=attack_class,
                                                         attack_params=attack_params, aug=aug, X=X, y=y,
                                                         batch_size=batch_size, transform=transform)
                if eps == 0:
                    data.append([i, model_t, eps, defended_accuracy[0], defended_confidence[0]])
                else:
                    data.append([i, model_t, eps, defended_adv_accuracy[0], defended_adv_confidence[0]])

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"{save_name}_{attack_class.__name__}.csv", index=False)


def run_experiment2(save_name="Experiment2", precompute_data=True, batch_size=100, n_runs=5, attack_class=PGDL2):
    print(f"Running Experiment 2 and saving to {save_name}_{attack_class.__name__}.csv")
    columns = ["Run", "Model", "Adversarial", "Augmentation", "Intensity", "Metric", "Confidence"]
    data = []
    models = ["U", "A"]
    augmentation_set = ["Noise", "Rotation", "Translation"]

    dataset = ds.CIFAR10
    transform = get_torchvision_dataset(dataset_class=dataset, train=False).transform.transforms[2]

    attack_class = PGDL2
    attack_params = {"eps": 2}

    for i in tqdm(range(n_runs)):
        X = None
        y = None
        if precompute_data:
            X, y = get_torchvision_dataset_sample(dataset_class=dataset, batch_size=batch_size)
        for model_t in models:
            if model_t in ["U"]:
                model = get_model(CIFAR10, aug=False)
            else:
                model = get_model(CIFAR10, aug=True)
            for augmentation in augmentation_set:
                if augmentation == "Noise":
                    for eps in [0.01, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 25]:
                        aug = augmentations.ExactL2Noise(eps=eps)
                        clean_accuracy, adv_accuracy, defended_accuracy, defended_adv_accuracy, \
                            defended_adv_robust, confidence, adv_confidence, defended_confidence, \
                            defended_adv_confidence = experiment(model=model, attack_class=attack_class,
                                                                 attack_params=attack_params, aug=aug, X=X, y=y,
                                                                 batch_size=batch_size, transform=transform)
                        data.append([i, model_t, False, augmentation, eps, defended_accuracy[0], confidence[0]])
                        data.append([i, model_t, True, augmentation, eps, defended_adv_robust[0], adv_confidence[0]])
                elif augmentation == "Rotation":
                    for degrees in [5, 10, 15, 20, 35, 45, 55, 65, 75, 90]:
                        aug = kornia.augmentation.RandomAffine(degrees=degrees, p=1)
                        clean_accuracy, adv_accuracy, defended_accuracy, defended_adv_accuracy, \
                            defended_adv_robust, confidence, adv_confidence, defended_confidence, \
                            defended_adv_confidence = experiment(model=model, attack_class=attack_class,
                                                                 attack_params=attack_params, aug=aug, X=X, y=y,
                                                                 batch_size=batch_size, transform=transform)
                        data.append([i, model_t, False, augmentation, degrees, defended_accuracy[0], confidence[0]])
                        data.append([i, model_t, True, augmentation, degrees, defended_adv_robust[0], adv_confidence[0]])
                elif augmentation == "Translation":
                    for translate in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                        aug = kornia.augmentation.RandomAffine(degrees=1, translate=(translate, translate), p=1)
                        clean_accuracy, adv_accuracy, defended_accuracy, defended_adv_accuracy, \
                            defended_adv_robust, confidence, adv_confidence, defended_confidence, \
                            defended_adv_confidence = experiment(model=model, attack_class=attack_class,
                                                                 attack_params=attack_params, aug=aug, X=X, y=y,
                                                                 batch_size=batch_size, transform=transform)
                        data.append([i, model_t, False, augmentation, translate, defended_accuracy[0], confidence[0]])
                        data.append([i, model_t, True, augmentation, translate, defended_adv_robust[0], adv_confidence[0]])

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(f"{save_name}_{attack_class.__name__}.csv", index=False)


if __name__ == "__main__":
    device = utils.Parameters.device
    run_experiment2()