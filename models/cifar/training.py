from datasets import get_torchvision_dataset_sample, get_torchvision_dataset, BatchNormalize, InverseNormalize
from torchvision.datasets import CIFAR10
from utils import show_grid
import kornia
import torch.nn as nn

dset = get_torchvision_dataset(CIFAR10)
bn = BatchNormalize(normalize_transform=dset.transform.transforms[2])
inv = InverseNormalize(normalize_transform=dset.transform.transforms[2])
X, y = get_torchvision_dataset_sample(CIFAR10)
X_inv = inv(X)
rn = kornia.augmentation.RandomRotation(degrees=90)
rn = kornia.augmentation.ColorJiggle(0.2, 0.3, 0.2, 0.3)
misc = nn.Sequential([
   kornia.augmentation.RandomAffine(360),
   kornia.augmentation.ColorJiggle(0.2, 0.3, 0.2, 0.3)
])
X_1_inv = misc(X_inv)
X_1 = bn(X_inv)
show_grid([[X[0], X_1[0]], [X_inv[0], X_1_inv[0]]])
