import torchvision.models as models
import torch
import utils
from torchvision.datasets import ImageNet
from datasets.image_datasets import get_torchvision_dataset
from attacks.image_attacks import ImagePerturbAttack, ImageSpatialAttack
import utils


model = models.resnet18(pretrained=True).eval().cuda()
dataset = get_torchvision_dataset(ImageNet)
no_samples = 5
indices = torch.randint(low=0, high=len(dataset)-1, size=(no_samples,))
X, y = dataset[indices]
X = X.cuda()
y = y.cuda()


from foolbox.attacks import LinfPGD
from foolbox import accuracy
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
attack = ImagePerturbAttack(foolbox_attack_class=LinfPGD)

print(f"{(model(X).argmax(-1) == y).float().mean()} clean accuracy")
advs, adv_preds, success, robust_accuracy = attack(model=model, input_batch=X, true_labels=y)
utils.show_img(advs[0][0], title="Adversarial Example")
utils.show_grid([advs[0][0], X[0]], title="Adversarial", captions=[adv_preds[0][0].item(), model(X)[0].argmax().item()])

attack = ImageSpatialAttack()
attack(model, input_batch=X, true_labels=y)
