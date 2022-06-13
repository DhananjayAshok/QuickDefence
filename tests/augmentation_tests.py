from augmentations.ImageAugmentation import *

import torchvision.datasets as datasets
from datasets.image_datasets import get_torchvision_dataset
from utils import show_grid


dset = get_torchvision_dataset(datasets.FashionMNIST)
noise = Noise(dist_params={"loc": 0, "scale": 0.001})
trans = (-0.1, 0.1)
affine = Affine(trans_x=trans, trans_y=trans)
img = dset[0][0]
img1 = noise(img)
img2 = affine(img)
captions = ["Original", "Noise", "Affine"]

show_grid(imgs=[img, img1, img2], captions=captions)
