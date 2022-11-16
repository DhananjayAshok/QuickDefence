import __init__ # Allow executation of this file as script from parent folder
import torch
import torch.nn as nn
from torchvision import models
from datasets import get_torchvision_dataset
from tqdm import tqdm
from datasets import InverseNormalize, BatchNormalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset_class, n_classes=10, input_channels=3,
          augmentation=lambda x: x, num_epochs=5, batch_size=32, lr=0.001, transform_location=2):
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

    model = models.resnet18(pretrained=True)
    if not input_channels == 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    if not n_classes == 1000:
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(num_epochs):
        for i, (imgs , labels) in tqdm(enumerate(train_loader), total=n_total_step):
            imgs = imgs.to(device)
            imgs = inverse_transform(imgs)
            imgs = augmentation(imgs)
            imgs = batch_transform(imgs)
            labels = labels.to(device)

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i+1) % 250 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}%')
            del imgs

    with torch.no_grad():
        number_corrects = 0
        number_samples = 0
        for i, (test_images_set, test_labels_set) in enumerate(test_loader):
            test_images_set = test_images_set.to(device)
            test_labels_set = test_labels_set.to(device)

            y_predicted = model(test_images_set)
            labels_predicted = y_predicted.argmax(axis=1)
            number_corrects += (labels_predicted == test_labels_set).sum().item()
            number_samples += test_labels_set.size(0)
        print(f'Overall accuracy {(number_corrects / number_samples) * 100} %')

    torch.save(model, f'models/{dataset_class.__name__}.pth')


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10, Caltech101, MNIST
    for dataset in [CIFAR10, Caltech101, MNIST]:
        input_channels = 10
        n_classes = 10
        n_epochs = 10
        if dataset == Caltech101:
            n_classes = 101
        if dataset == MNIST:
            input_channels = 1
            n_epochs = 20
        train(dataset_class=dataset, input_channels=input_channels, n_classes=n_classes, num_epochs=n_epochs)
