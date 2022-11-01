import torchvision.models as models
import torch
import utils
from torchvision.datasets import ImageNet, CIFAR10
from datasets.image_datasets import get_torchvision_dataset_sample, InverseNormalize
from attacks.image_attacks import ImagePerturbAttack, ImageSpatialAttack
import utils
from models import cifar
from datasets.image_datasets import get_torchvision_dataset_sample
from foolbox.attacks import LinfPGD
from adversarial_density import image_defence_density
import torch.multiprocessing as mp


if __name__ == "__main__":
    mp.set_start_method('spawn')

    model = cifar.get_model().cuda().eval()
    no_samples = 32
    X, y = get_torchvision_dataset_sample(CIFAR10, batch_size=no_samples)
    X, y = X.cuda(), y.cuda()
    inverse_transform = InverseNormalize(means=(0.4914, 0.4822, 0.4465), stds=(0.2023, 0.1994, 0.2010))
    image_X = inverse_transform(X[0])
    pred = model(X).argmin(-1)


    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    attack = ImagePerturbAttack(foolbox_attack_class=LinfPGD)
    advs = attack(model=model, input_batch=X, true_labels=y, preprocessing=preprocessing)
    adv_pred = model(advs).argmin(-1)
    accuracy, robust_accuracy, conditional_robust_accuracy, robustness, success = utils.get_attack_success_measures(model, inps=X, advs=advs, true_labels=y)
    print(accuracy, robust_accuracy, conditional_robust_accuracy, robustness)
    utils.show_grid([advs[0], image_X], title="Adversarial", captions=[pred[0], adv_pred[0]])
    print(image_defence_density(model.cpu(), advs[0].cpu(), y[0], n_samples=10, n_workers=2))

    attack = ImageSpatialAttack()
    advs = attack(model, input_batch=X, true_labels=y, preprocessing=preprocessing)
    adv_pred = model(advs).argmin(-1)
    accuracy, robust_accuracy, conditional_robust_accuracy, robustness, success = utils.get_attack_success_measures(model, inps=X, advs=advs, true_labels=y)
    print(accuracy, robust_accuracy, conditional_robust_accuracy, robustness)
    utils.show_grid([advs[0], image_X], title="Adversarial", captions=[pred[0], adv_pred[0]])


