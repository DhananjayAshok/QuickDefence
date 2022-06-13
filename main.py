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
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
attack = ImagePerturbAttack(foolbox_attack_class=LinfPGD)

print(f"{(model(X).argmax(-1) == y).float().mean()} clean accuracy")
advs, adv_preds, success, robust_accuracy = attack(model=model, input_batch=X, true_labels=y)

attack = ImageSpatialAttack()
attack(model, input_batch=X, true_labels=y)

adv_exs, input_batches, true_labels, a_preds = utils.get_adv_success(advs, success, X, y, adv_preds)

adv_ex, ib, tl, ap = adv_exs[1][0], input_batches[1][0], true_labels[1][0], a_preds[1][0]
utils.show_img(adv_ex, title="Adversarial Example")
utils.show_grid([adv_ex, ib], title="Adversarial", captions=[ap.item(),
                                                             model(ib.unsqueeze(0)).argmax(-1).item()])

from adversarial_density import image_defence_density
defence = lambda x: torch.rand(size=x.size()).to(x.device) + x
print(image_defence_density(model, adv_image=adv_ex, true_label=tl, defence=defence))

