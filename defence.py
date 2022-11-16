"""
Test Time Wrapper Around Defended Network
"""
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import utils


class DefendedNetwork(nn.Module):
    def __init__(
        self,
        network,
        data_augmentation,
        sample_rate=10,
        aggregation="mean",
        data_n_dims=None,
        output_shape=(),
        transform=lambda x: x,
        inverse_transform=lambda x: x,
    ):
        """

        :param network:
        :param data_augmentation:
        :param sample_rate:
        :param aggregation:
        :param n_workers: for multiprocessing
        :param data_n_dims: the len(x.shape) where x is a single datapoint no batch dimension.
        :param output_shape: the o.shape where o is a single datapoint output from model no batch dimension
        :param transform: BatchNormalize Object that takes raw dataset input tensor to the format the network accepts
            (not input image etc)
        :param inverse_transform: InverseNormalize that takes the tensor the network accepts and undoes all the procedures done
         by the above transform

         One example of tranform and inverse transform is if
         image_transform = torchvision.transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ) ]) is the transform used to load the images: then

        transform = transforms.Compose([
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) ) ])
        inverse_transform = transforms.Compose([
            transforms.Normalize((0.0, 0.0, 0.0), (1/2.023, 1/0.1994, 1/0.2010)),
            transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1)) ])
        """
        nn.Module.__init__(self)
        self.network = network.eval()
        self.sample_rate = sample_rate
        self.data_augmentation = data_augmentation
        self.aggregation = aggregation
        self.data_n_dims = data_n_dims
        self.output_shape = output_shape
        self.transform = transform
        self.inverse_transform = inverse_transform

    def aggregate_logits(self, logits):
        if self.aggregation == "mean":
            avg_logits = torch.sum(logits, dim=1) / self.sample_rate
            return avg_logits
        else:
            raise ValueError(f"Aggregation method {self.aggregation} not implemented")

    def forward(self, x):
        """

        :param x: if data_n_dims was specified during initialization then x can be batch or single,
            else must be single datapoint (but can have batch dimension)
        :return:
        """
        batch_size = x.shape[0]
        with torch.no_grad():
            x = self.inverse_transform(x)
            x = utils.repeat_batch_images(x, num_repeat=self.sample_rate)
            x = self.data_augmentation(x)
            x = self.transform(x)
            logits = self.network(x)
            logits = logits.view(batch_size, self.sample_rate, logits.shape[1])
        return self.aggregate_logits(logits)

    def generate_input(self, x):
        x = self.data_augmentation(x)
        return x

    def get_model(self):
        return self.network

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(-1)
