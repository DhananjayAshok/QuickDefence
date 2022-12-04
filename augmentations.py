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





class ExactLinfNoise:
    def __init__(self, eps, temp=1):
        self.eps = eps
        self.temp = temp

    def __call__(self, x):
        val = self.eps * self.temp
        n = ((torch.rand(x.size())/(2*val)) - val).to(x.device)
        return x+n


class ExactL2Noise:
    def __init__(self, eps, temp=1):
        self.eps = eps
        self.temp = temp

    def __call__(self, x):
        val = self.eps * self.temp
        n = torch.rand(x.size()) - 0.5
        n_norm = n.view(x.shape[0], -1).norm(dim=1).max()
        n = n / n_norm
        n = (n * val).to(x.device)
        return x+n


class CIFARAugmentation:
    noise = ExactL2Noise(eps=5)
    rot = kornia.augmentation.RandomAffine(degrees=(-35, 35), p=1.0)
    trans = kornia.augmentation.RandomAffine(degrees=1, translate=(0.2, 0.2), p=1.0)
    standard_aug = DataAugmentation([noise, rot, trans])


class CaltechAugmentation:
    noise = kornia.augmentation.RandomGaussianNoise(std=0.1, p=0.6)
    color = kornia.augmentation.ColorJiggle(brightness=(0.75, 1.1), contrast=(1, 2), hue=(-0.4, 0.4),
                                            saturation=(0, 6),
                                            p=0.3)
    affine = kornia.augmentation.RandomAffine(degrees=(-1, 1), translate=(0, 0.25), p=1.0)
    standard_aug = DataAugmentation([noise, color, affine])


class MNISTAugmentation:
    noise = kornia.augmentation.RandomGaussianNoise(std=0.005, p=0.3)
    affine = kornia.augmentation.RandomAffine(degrees=(-1, 1), translate=(0, 0.35), p=1.0)
    standard_aug = DataAugmentation([noise, affine])



def get_augmentation(dset_class, variant=None):
    if variant is None:
        if dset_class == ds.CIFAR10:
            return CIFARAugmentation.standard_aug
        if dset_class == ds.Caltech101:
            return CaltechAugmentation.standard_aug
        if dset_class == ds.MNIST:
            return MNISTAugmentation.standard_aug
        else:
            return MNISTAugmentation.standard_aug
    else:
        if variant == "translation":
            if dset_class == ds.CIFAR10:
                return CIFARAugmentation.affine
            if dset_class == ds.Caltech101:
                return CaltechAugmentation.affine
            if dset_class == ds.MNIST:
                return MNISTAugmentation.affine
            else:
                return MNISTAugmentation.affine

