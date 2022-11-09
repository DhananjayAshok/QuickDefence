from torchvision.datasets import ImageNet, CIFAR10
import torchvision.transforms as transforms

from datasets.image_datasets import get_torchvision_dataset_sample, InverseNormalize
import utils
from augmentations import ImageAugmentation as ia


def test_noise(dataset=CIFAR10, param_sets=[{"dist": "gaussian", "loc": 0, "scale": 0.0001}]):
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

