import __init__  # Allow executation of this file as script from parent folder
import matplotlib.pyplot as plt
import kornia


import utils
from datasets import (
    BatchNormalize,
    InverseNormalize,
    get_torchvision_dataset,
    get_torchvision_dataset_sample,
)
from defence import DefendedNetwork


def test_augmentation_visual(Aug_class, dataset, param_sets=[], batch_size=5):
    device = utils.Parameters.device
    transform = get_torchvision_dataset(dataset).transform.transforms[2]
    X, y = get_torchvision_dataset_sample(dataset, batch_size=batch_size)
    X, y = X.to(device), y.to(device)

    inverse_transform = InverseNormalize(normalize_transform=transform)
    for param_set in param_sets:
        aug = Aug_class(**param_set)
        image_X = inverse_transform(X)
        auged_X = aug(image_X)
        imgs = []
        for i in range(batch_size):
            imgs.append([image_X[i], auged_X[i]])
        utils.show_grid(
            imgs,
            title=f"Noise on {dataset.__name__}|({param_set})",
            captions=[["Image", "Augmented Image"] for i in range(batch_size)],
        )


def dict_str(d):
    s = ""
    for k in d:
        s += f"|{k}:{d[k]}|"


def test_augmentation_defence(
    model,
    Aug_class,
    dataset,
    param_sets=[],
    output_shape=(10,),
    sample_rate=10,
    batch_size=50,
):
    device = utils.Parameters.device
    X, y = get_torchvision_dataset_sample(dataset, batch_size=batch_size)
    X, y = X.to(device), y.to(device)
    accuracies = [utils.get_accuracy_logits(y, model(X))]
    param_list = ["None"]
    transform = get_torchvision_dataset(dataset).transform.transforms[2]
    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)
    for params in param_sets:
        aug = Aug_class(**params)
        param_list.append(str(params))
        defended = DefendedNetwork(
            network=model,
            data_augmentation=aug,
            data_n_dims=3,
            transform=batch_transform,
            inverse_transform=inverse_transform,
            output_shape=output_shape,
            sample_rate=sample_rate,
        )
        accuracies.append(utils.get_accuracy_logits(y, defended(X)))
    plt.plot(param_list, accuracies)
    plt.show()


noiseClass = kornia.augmentation.RandomGaussianNoise#std=0.05, p=1.0)
cjClass = kornia.augmentation.ColorJiggle#(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.3)
affineClass = kornia.augmentation.RandomAffine#(degrees=(-20, 20), translate=(0, 0.2), p=1)
zero_1 = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.4, 0.5, 0.75, 0.8, 0.9, 1]
noise_param_set = [{"std": i, "p": 1} for i in zero_1]
brightness_param_set = [{"brightness": i, "p": 1} for i in zero_1]
contrast_param_set = [{"contrast": i, "p": 1} for i in [0.1, 0.5, 1, 2, 5, 10, 15, 20]]
saturation_param_set = [{"saturation": i, "p": 1} for i in [0.1, 0.4]]
hue_param_set = [{"hue": (-i, i), "p": 1} for i in [0.1, 0.2, 0.3, 0.4]]
color_param_set = brightness_param_set + contrast_param_set + saturation_param_set + hue_param_set
rotation_param_set = [{"degrees": (-i, i), "p": 1} for i in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]]
translation_param_set = [{"degrees": (-i, i), "translate": (0, i), "p": 1} for i in zero_1]
affine_param_set = rotation_param_set+translation_param_set


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10, Caltech101, MNIST
    for dataset in [MNIST]:
        test_augmentation_visual(noiseClass, dataset, param_sets=noise_param_set)
        test_augmentation_visual(affineClass, dataset, param_sets=affine_param_set)
        if dataset in [MNIST]:
            continue
        else:
            test_augmentation_visual(cjClass, dataset, param_sets=color_param_set)
