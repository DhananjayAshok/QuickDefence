from augmentations.ImageAugmentation import *

import torchvision.datasets as datasets
from datasets.image_datasets import get_torchvision_dataset
from utils import show_grid


cifar = get_torchvision_dataset(datasets.CIFAR10)
noise = Noise(dist_params={"loc": 0, "scale": 0.01})
affine = Affine(trans_x=(-0.05, 0.05))
img = cifar[0][0]
img1 = noise(img)
img2 = affine(img)
captions = ["Original", "Noise", "Affine"]

show_grid(imgs=[img, img1, img2], captions=captions)
