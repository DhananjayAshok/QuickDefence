import kornia
import torch.nn as nn
import torch


class DataAugmentation(nn.Module):
    def __init__(self, auglist):
       super().__init__()
       self.auglist = auglist

    @torch.no_grad()
    def forward(self, x):
        for aug in self.auglist:
            x = aug(x)
        return x


noise = kornia.augmentation.RandomGaussianNoise(std=0.05, p=1.0)
color = kornia.augmentation.ColorJiggle(brightness=(0.9, 1.1), contrast=(1, 2), hue=(-0.25, 0.25), saturation=(0, 5),
                                        p=1.0)
affine = kornia.augmentation.RandomAffine(degrees=(-20, 20), translate=(0, 0.2), p=1)

standard_cifar_aug = DataAugmentation([noise, color, affine])
