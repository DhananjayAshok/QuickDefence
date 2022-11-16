import kornia
import torch.nn as nn
import torch
import torchvision.datasets as ds

class DataAugmentation(nn.Module):
    def __init__(self, auglist):
       super().__init__()
       self.auglist = auglist

    @torch.no_grad()
    def forward(self, x):
        for aug in self.auglist:
            x = aug(x)
        return x

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.auglist)


class CIFARAugmentation:
    noise = kornia.augmentation.RandomGaussianNoise(std=0.05, p=1.0)
    color = kornia.augmentation.ColorJiggle(brightness=(0.9, 1.1), contrast=(1, 2), hue=(-0.25, 0.25), saturation=(0, 5),
                                            p=1.0)
    affine = kornia.augmentation.RandomAffine(degrees=(-20, 20), translate=(0, 0.2), p=1)
    standard_aug = DataAugmentation([noise, color, affine])


class CaltechAugmentation:
    noise = kornia.augmentation.RandomGaussianNoise(std=0.05, p=1.0)
    color = kornia.augmentation.ColorJiggle(brightness=(0.9, 1.1), contrast=(1, 2), hue=(-0.25, 0.25),
                                            saturation=(0, 5),
                                            p=1.0)
    affine = kornia.augmentation.RandomAffine(degrees=(-20, 20), translate=(0, 0.2), p=1)
    standard_aug = DataAugmentation([noise, color, affine])


class MNISTAugmentation:
    noise = kornia.augmentation.RandomGaussianNoise(std=0.05, p=1.0)
    affine = kornia.augmentation.RandomAffine(degrees=(-20, 20), translate=(0, 0.2), p=1)
    standard_aug = DataAugmentation([noise, affine])


def get_augmentation(dset_class):
    if dset_class == ds.CIFAR10:
        return CIFARAugmentation.standard_aug
    if dset_class == ds.Caltech101:
        return CaltechAugmentation.standard_aug
    if dset_class == ds.MNIST:
        return MNISTAugmentation.standard_aug
    else:
        return MNISTAugmentation.standard_aug
