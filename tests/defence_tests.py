import __init__  # Allow executation of this file as script from parent folder
from torchvision.datasets import CIFAR10
import utils
from augmentations import get_augmentation
from datasets import InverseNormalize, get_torchvision_dataset_sample, BatchNormalize, get_torchvision_dataset, \
    get_index_to_class
from defence import DefendedNetwork
from models import get_model


def test_defence_accuracy(dataset_class=CIFAR10, output_shape=(10, ), no_samples=32, aug=None,
                          sample_rate=10, show=True):
    transform = get_torchvision_dataset(dataset_class, train=False).transform.transforms[2]
    index_to_class = get_index_to_class(dataset_class=dataset_class)
    model = get_model(dataset_class)

    X, y = get_torchvision_dataset_sample(dataset_class, batch_size=no_samples)
    pred = model(X)

    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)
    preprocessing = utils.normalize_to_dict(transform)
    if aug is None:
        aug = get_augmentation(dataset_class)
    image_X = inverse_transform(X)
    auged_X = aug(image_X)
    auged_pred = model(inverse_transform(auged_X))
    defended = DefendedNetwork(
        network=model,
        data_augmentation=aug,
        data_n_dims=3,
        transform=batch_transform,
        inverse_transform=inverse_transform,
        output_shape=output_shape,
        sample_rate=sample_rate,
    )
    defended_pred = defended.predict(X)
    print(
        f"Clean Accuracy {utils.get_accuracy_logits(y, pred)} vs Defended Accuracy "
        f"{utils.get_accuracy(y, defended_pred)}"
    )
    if show:
        utils.show_adversary_vs_original_with_preds(auged_X, img_X=image_X, y=y, adv_pred=auged_pred, pred=pred,
                                                    defended_pred=defended_pred, index_to_class=index_to_class,
                                                    n_show=5)


if __name__ == "__main__":
    import torchvision.datasets as ds
    device = utils.Parameters.device
    test_defence_accuracy(dataset_class=ds.Caltech101, output_shape=(101,))