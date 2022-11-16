import kornia
import torch.nn as nn
import torch


noise = kornia.augmentation.RandomGaussianNoise(std=0.1, p=1.0)
color = kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)


class DataAugmentation(nn.Module):
    def __init__(self, auglist):
       super().__init__()
       self.auglist = auglist

    @torch.no_grad()
    def forward(self, x):
        for aug in self.auglist:
            x = aug(x)
        return x
