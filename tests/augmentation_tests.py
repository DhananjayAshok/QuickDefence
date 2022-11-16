from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from datasets import get_torchvision_dataset_sample, InverseNormalize
import utils
from augmentations import ImageAugmentation as ia
from defence import DefendedNetwork


def test_noise_visual(dataset=CIFAR10, param_sets=[{"dist": "gaussian", "loc": 0, "scale": 0.01}]):
    device = utils.Parameters.device
    no_samples = 5
    X, y = get_torchvision_dataset_sample(dataset, batch_size=no_samples)
    X, y = X.to(device), y.to(device)

    transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    inverse_transform = InverseNormalize(normalize_transform=transform)
    for param_set in param_sets:
        aug = ia.Noise(dist=param_set["dist"], dist_params=param_set)
        imgs = []
        for i in range(no_samples):
            image_X = inverse_transform(X[i])
            auged_X = aug(image_X)
            imgs.append([image_X, auged_X])
        utils.show_grid(imgs, title=f"Noise on {dataset.__name__}|({param_set})",
                        captions=[["Image", "Augmented Image"] for i in range(no_samples)])


def test_noise_defence_img(model, dataset=CIFAR10, dist="gaussian", scales=[], output_shape=(10, ), sample_rate=10):
    import matplotlib.pyplot as plt
    device = utils.Parameters.device
    no_samples = 50
    X, y = get_torchvision_dataset_sample(dataset, batch_size=no_samples)
    X, y = X.to(device), y.to(device)
    accuracies = [utils.get_accuracy_logits(y, model(X))]
    transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    inverse_transform = InverseNormalize(normalize_transform=transform)
    for scale in scales:
        aug = ia.Noise(dist=dist, dist_params={"scale": scale})
        defended = DefendedNetwork(network=model, data_augmentation=aug, data_n_dims=3, transform=transform,
                                   inverse_transform=inverse_transform, output_shape=output_shape,
                                   sample_rate=sample_rate)
        accuracies.append(utils.get_accuracy_logits(y, defended(X)))
    plt.plot([0]+scales, accuracies)
    plt.show()
