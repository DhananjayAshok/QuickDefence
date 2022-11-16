from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from datasets import get_torchvision_dataset, get_torchvision_dataset_sample, InverseNormalize, BatchNormalize
import utils
from defence import DefendedNetwork
import matplotlib.pyplot as plt


def test_augmentation_visual(Aug_class, dataset=CIFAR10, param_sets=[], batch_size=5):
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
        utils.show_grid(imgs, title=f"Noise on {dataset.__name__}|({param_set})",
                        captions=[["Image", "Augmented Image"] for i in range(batch_size)])


def dict_str(d):
    s = ""
    for k in d:
        s += f"|{k}:{d[k]}|"


def test_augmentation_defence(model, Aug_class, dataset=CIFAR10, param_sets=[], output_shape=(10, ), sample_rate=10, batch_size=50):
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
        defended = DefendedNetwork(network=model, data_augmentation=aug, data_n_dims=3, transform=batch_transform,
                                   inverse_transform=inverse_transform, output_shape=output_shape,
                                   sample_rate=sample_rate)
        accuracies.append(utils.get_accuracy_logits(y, defended(X)))
    plt.plot(param_list, accuracies)
    plt.show()
