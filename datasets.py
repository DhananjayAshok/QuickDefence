import torchvision.datasets as ds
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from utils import safe_mkdir

data_root = "data/"


def sample_torch_dataset(dset, batch_size=32, shuffle=False):
    f, h = dset[0]
    X_shape = f.shape
    # We assume y is a scalar output
    batch_shape = (batch_size, ) + X_shape
    y_shape = (batch_size, )
    device = f.device
    X = torch.zeros(size=batch_shape).to(device)
    y = torch.zeros(size=y_shape).to(device)
    if shuffle:
        idx = torch.randint(low=0, high=len(dset), size=(batch_size, ))
    else:
        idx = range(batch_size)
    for i, id in enumerate(idx):
        X[i], y[i] = dset[id]
    return X, y


def get_torchvision_dataset(dataset_class, train=False):
    transform = None
    if dataset_class == ds.CIFAR10:
        transform = transforms.Compose(
            [
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    if dataset_class == ds.Caltech101:
        transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            BatchNormalize(means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    if dataset_class == ds.MNIST:
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
            BatchNormalize(means=(0.127,), stds=(0.2959,)),
        ])
    save_path = data_root + f"/{dataset_class.__name__}/"
    safe_mkdir(save_path)
    if not dataset_class == ds.Caltech101:
        return dataset_class(save_path, transform=transform, download=True, train=train)
    else:
        return dataset_class(save_path, transform=transform, download=True)


def get_torchvision_dataset_sample(dataset_class, train=False, batch_size=32):
    dset = get_torchvision_dataset(dataset_class, train=train)
    X, y = sample_torch_dataset(dset, batch_size=batch_size, shuffle=True)
    return X, y.long()


class BatchNormalize(nn.Module):
    def __init__(self, means=None, stds=None, normalize_transform=None):
        assert normalize_transform is not None or (
                means is not None and stds is not None
        )
        nn.Module.__init__(self)
        if normalize_transform is not None:
            means = normalize_transform.mean
            stds = normalize_transform.std
        self.op = transforms.Normalize(mean=means, std=stds)
        self.mean = means
        self.std = stds
        self.tensors = False

    def __call__(self, x):
        ndims = len(x.shape)
        if ndims == 3:
            channels = len(self.op.mean)
            if x.shape[0] == channels:
                return self.op(x)
            elif x.shape[0] < channels: # Assume this means its 1
                #assert x.shape[0] == 1
                in_x = torch.cat([x for i in range(channels)], dim=0)
                return self.op(in_x)
            else:
                return self.op(x[:channels])
        else:
            return self.batch_call(x)

    def batch_call(self, x):
        if not self.tensors:
            self.std = torch.Tensor(self.std).to(x.device)
            self.mean = torch.Tensor(self.mean).to(x.device)
            self.tensors = True
        bs, c, h, w = x.shape
        x = x.view(bs, c, h * w) - self.mean.expand(bs, c).unsqueeze(2)
        x = x / self.std.expand(bs, c).unsqueeze(2)
        return x.view(bs, c, h, w)

    def __repr__(self):
        return self.op.__repr__()


class InverseNormalize(nn.Module):
    def __init__(self, means=None, stds=None, normalize_transform=None):
        assert normalize_transform is not None or (
            means is not None and stds is not None
        )
        nn.Module.__init__(self)
        if normalize_transform is not None:
            means = normalize_transform.mean
            stds = normalize_transform.std
        inverse_stds = [1 / s for s in stds]
        inverse_means = [-m for m in means]
        default_means = [0.0 for s in stds]
        default_stds = [1.0 for m in means]
        self.trans = transforms.Compose(
            [
                transforms.Normalize(default_means, inverse_stds),
                transforms.Normalize(inverse_means, default_stds),
            ]
        )
        self.std = None
        self.mean = None

    def __call__(self, x):
        ndims = len(x.shape)
        if ndims == 3:
            return self.trans(x)
        else:
            return self.batch_call(x)

    def batch_call(self, x):
        if self.std is None:
            self.std = torch.Tensor(self.trans.transforms[0].std).to(x.device)
            self.mean = torch.Tensor(self.trans.transforms[1].mean).to(x.device)
        bs, c, h, w = x.shape
        x = x.view(bs, c, h * w) / self.std.expand(bs, c).unsqueeze(2)
        x = x - self.mean.expand(bs, c).unsqueeze(2)
        return x.view(bs, c, h, w)

    def __repr__(self):
        return self.trans.__repr__()
