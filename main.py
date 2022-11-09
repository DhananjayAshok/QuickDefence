import torchvision.models as models
import torch
import utils
from torchvision.datasets import ImageNet, CIFAR10
from datasets.image_datasets import get_torchvision_dataset_sample, InverseNormalize
from attacks.image_attacks import FoolboxImageAttack
import utils
from models import cifar
from foolbox.attacks import LinfPGD
from adversarial_density import image_defence_density
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from augmentations import ImageAugmentation as ia
from defence import DefendedNetwork


if __name__ == "__main__":
    mp.set_start_method('spawn')
    device = utils.Parameters.device
    model = cifar.get_model().eval().to(device)
    no_samples = 32
    X, y = get_torchvision_dataset_sample(CIFAR10, batch_size=no_samples)
    X, y = X.to(device), y.to(device)
    pred = model(X)

    transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    inverse_transform = InverseNormalize(means=(0.4914, 0.4822, 0.4465), stds=(0.2023, 0.1994, 0.2010))
    aug = ia.Noise()
    image_X = inverse_transform(X[0])
    auged_X = aug(image_X)
    utils.show_grid([image_X, auged_X], title="Augmentation", captions=["Image", "Augmented Image"])
    defended = DefendedNetwork(network=model, data_augmentation=aug, data_n_dims=3, transform=transform,
                               inverse_transform=inverse_transform, output_shape=(10,), sample_rate=10)
    defended_pred = defended.predict(X)
    print(f"Clean Accuracy {utils.get_accuracy_logits(y, pred)} vs Defended Accuracy "
          f"{utils.get_accuracy(y, defended_pred)}")

    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    attack = FoolboxImageAttack(foolbox_attack_class=LinfPGD)
    advs = attack(model=model, input_batch=X, true_labels=y, preprocessing=preprocessing)
    adv_pred = model(advs).argmin(-1)
    accuracy, robust_accuracy, conditional_robust_accuracy, robustness, success = utils.get_attack_success_measures(model, inps=X, advs=advs, true_labels=y)
    print(accuracy, robust_accuracy, conditional_robust_accuracy, robustness)
    accuracy, robust_accuracy, conditional_robust_accuracy, robustness, success = utils.get_attack_success_measures(
        defended, inps=X, advs=advs, true_labels=y)
    print(accuracy, robust_accuracy, conditional_robust_accuracy, robustness)

    utils.show_grid([advs[0], image_X], title="Adversarial", captions=[pred[0].argmax(-1), f"{adv_pred[0], defended_pred[0]}"])
    print(image_defence_density(model.cpu(), advs[0].cpu(), y[0], n_samples=10, n_workers=2))

    """
    attack = ImageSpatialAttack()
    advs = attack(model, input_batch=X, true_labels=y, preprocessing=preprocessing)
    adv_pred = model(advs).argmin(-1)
    accuracy, robust_accuracy, conditional_robust_accuracy, robustness, success = utils.get_attack_success_measures(model, inps=X, advs=advs, true_labels=y)
    print(accuracy, robust_accuracy, conditional_robust_accuracy, robustness)
    utils.show_grid([advs[0], image_X], title="Adversarial", captions=[pred[0], adv_pred[0]])
    """