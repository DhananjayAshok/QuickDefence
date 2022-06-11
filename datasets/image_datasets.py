import torchvision.transforms

from datasets.Dataset import data_root
from utils import safe_mkdir


def get_torchvision_dataset(dataset_class):
    save_path = data_root+f"/{dataset_class.__name__}/"
    safe_mkdir(save_path)
    return dataset_class(save_path, transform=torchvision.transforms.ToTensor(), download=True)





