import os.path
import torch
import torchvision.transforms as transforms
import torchvision.datasets as ds
import pandas as pd
import numpy as np
from datasets import data_root, sample_torch_dataset
from utils import safe_mkdir
from PIL import Image


def get_torchvision_dataset(dataset_class, train=False):
    transform = None
    if dataset_class == ds.CIFAR10:
        transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        ])

    save_path = data_root+f"/{dataset_class.__name__}/"
    safe_mkdir(save_path)
    return dataset_class(save_path, train=train, transform=transform, download=True)


def get_torchvision_dataset_sample(dataset_class, train=False, batch_size=32):
    dset = get_torchvision_dataset(dataset_class, train=train)
    X, y = sample_torch_dataset(dset, batch_size=batch_size, shuffle=True)
    return X, y.long()


class InverseNormalize:
    def __init__(self, means=None, stds=None, normalize_transform=None):
        assert normalize_transform is not None or (means is not None and stds is not None)
        if normalize_transform is not None:
            means = normalize_transform.mean
            stds = normalize_transform.std
        inverse_stds = [1/s for s in stds]
        inverse_means = [-m for m in means]
        default_means = [0.0 for s in stds]
        default_stds = [1.0 for m in means]
        self.trans = transforms.Compose([
            transforms.Normalize(default_means, inverse_stds), transforms.Normalize(inverse_means, default_stds)])

    def __call__(self, x):
        return self.trans(x)

    def __repr__(self):
        return self.trans.__repr__()
