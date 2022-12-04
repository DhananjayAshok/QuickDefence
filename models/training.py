import torch
import torch.nn as nn
from torchvision import models

import augmentations
from datasets import get_torchvision_dataset
from tqdm import tqdm
from datasets import InverseNormalize, BatchNormalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset_class, n_classes=10, input_channels=3, n_samples=5, use_aug=False,
          augmentation=lambda x: x, num_epochs=20, batch_size=32, lr=0.001, transform_location=2, save_suffix=""):
    train_dataset = get_torchvision_dataset(dataset_class, train=True)
    test_dataset = get_torchvision_dataset(dataset_class, train=False)
    if transform_location > 0:
        transform = train_dataset.transform.transforms[transform_location]
    batch_transform = BatchNormalize(normalize_transform=transform)
    inverse_transform = InverseNormalize(normalize_transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset
        , batch_size=batch_size
        , shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset
        , batch_size=batch_size
        , shuffle=True)
    n_total_step = len(train_loader)
    print(n_total_step)
    if not use_aug:
        n_samples = 0

    model = models.resnet50(pretrained=True)
    if not input_channels == 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    if not n_classes == 1000:
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    for epoch in range(num_epochs):
        for i, (imgs , labels) in tqdm(enumerate(train_loader), total=n_total_step):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            inv_imgs = inverse_transform(imgs)
            n_correct_n = 0
            for j in range(n_samples):
                optimizer.zero_grad()
                aug_imgs = augmentation(inv_imgs)
                aug_imgs = batch_transform(aug_imgs)

                labels_hat = model(aug_imgs)
                n_correct_n = (labels_hat.argmax(axis=1) == labels).sum().item()
                loss_value = criterion(labels_hat, labels)
                loss_value.backward()
                optimizer.step()
                del aug_imgs
            if (i+1) % 250 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}, aug acc = {100*(n_correct_n/labels.size(0)):.2f}%')
            if (i+1) % 500 == 0:
                torch.save(model, f'models/{dataset_class.__name__}_{save_suffix}({epoch}_{i}).pth')
            del imgs
        scheduler.step()
        val_loss = _val(model, augmentation, test_loader, use_aug, criterion)
    torch.save(model, f'models/{dataset_class.__name__}_{save_suffix}.pth')


def _val(model, augmentation, test_loader, use_aug, loss):
    with torch.no_grad():
        number_corrects = 0
        number_samples = 0
        val_losses = []
        for i, (test_images_set, test_labels_set) in enumerate(test_loader):
            test_images_set = test_images_set.to(device)
            test_labels_set = test_labels_set.to(device)

            y_predicted = model(test_images_set)
            labels_predicted = y_predicted.argmax(axis=1)
            loss_value = loss(y_predicted, test_labels_set)
            number_corrects += (labels_predicted == test_labels_set).sum().item()
            number_samples += test_labels_set.size(0)
            val_losses.append(loss_value)
        print(f'Overall accuracy No Aug: {(number_corrects / number_samples) * 100} % Loss: {loss_value}')
        number_corrects = 0
        number_samples = 0
        if use_aug:
            for i, (test_images_set, test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.to(device)
                test_images_set = augmentation(test_images_set)
                test_labels_set = test_labels_set.to(device)

                y_predicted = model(test_images_set)
                labels_predicted = y_predicted.argmax(axis=1)
                number_corrects += (labels_predicted == test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy Aug: {(number_corrects / number_samples) * 100} %')
        return sum(val_losses)/len(val_losses)


def run_all():
    from torchvision.datasets import CIFAR10, Caltech101, MNIST
    for dataset in [CIFAR10]:
        for i in range(2):
            input_channels = 3
            n_classes = 10
            n_epochs = 10
            if i == 0:
                augmentation = augmentations.CIFARAugmentation.standard_aug
                suffix = "_aug"
                use_aug = True
            else:
                suffix = ""
                augmentation = lambda x: x
                use_aug = False
            train(dataset_class=dataset, input_channels=input_channels, n_classes=n_classes, num_epochs=n_epochs,
                  augmentation=augmentation, save_suffix=suffix, use_aug=use_aug)


if __name__ == "__main__":
    run_all()
