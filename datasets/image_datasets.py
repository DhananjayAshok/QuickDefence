import os.path

import torch
import torchvision.transforms
import torchvision.datasets as ds
import pandas as pd
import numpy as np
from datasets.Dataset import data_root
from utils import safe_mkdir
from PIL import Image


def get_torchvision_dataset(dataset_class):
    if dataset_class == ds.ImageNet:
        save_path = data_root + f"/{dataset_class.__name__}/tiny-imagenet-200/"
        return load_image_net()
    save_path = data_root+f"/{dataset_class.__name__}/"
    safe_mkdir(save_path)
    return dataset_class(save_path, transform=torchvision.transforms.ToTensor(), download=True)


def foolbox_tiny_image_net(batch_size=16):
    from foolbox import samples
    from datasets.Dataset import Dataset
    image, label = samples(fmodel=None,
                           bounds=(0, 1), dataset="imagenet", batchsize=batch_size, data_format="channels_first")
    image = torch.from_numpy(image)
    label = torch.from_numpy(label)
    return Dataset(X=image, y=label)


def get_no_string(i):
    return f'{i:08d}'


def load_image_net(batch_size=32):
    from datasets.Dataset import Dataset
    imnet = "ILSVRC2012"
    img_path = os.path.join(data_root, "Imagenet", f"{imnet}_img_val")
    gt_path = os.path.join(data_root, "Imagenet", f"{imnet}_validation_ground_truth.txt")
    ys = pd.read_csv(gt_path, header=None)[0]
    total_size = len(ys)
    idx = np.random.randint(0, total_size, size=(batch_size, ))
    label = torch.tensor(ys[idx].values)
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tensorify = torchvision.transforms.ToTensor()
    preprocessing = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        tensorify,
        normalize,
    ])
    image = torch.zeros(size=(batch_size, 3, 224, 224))
    for i, id in enumerate(idx):
        pil_image = Image.open(f"{img_path}/{imnet}_val_{get_no_string(id)}.JPEG").convert('RGB')
        image[i] = preprocessing(pil_image)
    return Dataset(X=image, y=label)


def load_tiny_image_net(path, label_limit=10):
    import os
    from PIL import Image
    import torch
    from datasets.Dataset import Dataset
    from tqdm import tqdm
    labels = []
    images = []
    trans = torchvision.transforms.ToTensor()
    print(f"Loading Tiny ImageNet")
    for i, label in tqdm(enumerate(os.listdir(f"{path}/train/"))):
        if label_limit is not None and i > label_limit:
            break
        for image_name in os.listdir(f"{path}/train/{label}/images"):
            p = f"{path}/train/{label}/images/{image_name}"
            img = Image.open(p)
            img = trans(img)
            if img.shape == (1, 64, 64):
                continue
            images.append(img.reshape(1, 3, 64, 64))
            labels.append(torch.LongTensor([i]))
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return Dataset(X=images, y=labels)









