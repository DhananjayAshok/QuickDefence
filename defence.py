"""
Test Time Wrapper Around Defended Network
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp


class DefendedNetwork(nn.Module):
    def __init__(self, network, data_augmentation, sample_rate=10, aggregation="mean", n_workers=1, data_n_dims=None,
                 output_shape=(), transform=lambda x: x, inverse_transform=lambda x: x):
        """

        :param network:
        :param data_augmentation:
        :param sample_rate:
        :param aggregation:
        :param n_workers: for multiprocessing
        :param data_n_dims: the len(x.shape) where x is a single datapoint no batch dimension.
        :param output_shape: the o.shape where o is a single datapoint output from model no batch dimension
        :param transform: transform that takes raw dataset input tensor to the format the network accepts
            (not input image etc)
        :param inverse_transform: transform that takes the tensor the network accepts and undoes all the procedures done
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
        self.network = network
        self.sample_rate = sample_rate
        self.data_augmentation = data_augmentation
        self.aggregation = aggregation
        self.n_workers = n_workers
        self.data_n_dims = data_n_dims
        self.output_shape = output_shape
        self.transform = transform
        self.inverse_transform = inverse_transform

    def aggregate_outputs(self, outputs):
        if self.aggregation == "mean":
            return sum(outputs) / len(outputs)
        return sum(outputs) / len(outputs)

    def forward_single_sample(self, x):
        """

        :param x: input tensor without any batch dimension
        :return: output tensor of model run on one augmented input
        """
        x = self.inverse_transform(x)
        input = self.generate_input(x)
        input = self.transform(input)
        input = input[None, :]  # Make it have a batch axis
        out = self.network(input)
        return out

    def forward_no_batch(self, x, n_workers=None):
        if n_workers is None:
            n_workers = self.n_workers
        xs = [x for i in range(self.sample_rate)]
        if n_workers == 1:
            outputs = [self.forward_single_sample(x) for x in xs]
        else:
            pool = mp.pool(processes=n_workers)
            outputs = pool.map(self.forward_single_sample, xs)

        return self.aggregate_outputs(outputs)

    def forward(self, x, n_workers=None):
        """

        :param x: if data_n_dims was specified during initialization then x can be batch or single,
            else must be single datapoint (but can have batch dimension)
        :param n_workers: number of cores for multiprocessing if not specified uses initialized value
        :return:
        """
        if self.data_n_dims is None or len(x.shape) == self.data_n_dims:
            return self.forward_no_batch(x, n_workers=n_workers)
        else:
            batch_size = x.shape[0]
            out_shape = (batch_size, ) + self.output_shape
            out = torch.zeros(out_shape).to(x.device)
            for b in batch_size:
                out[b] = self.forward_no_batch(x[b], n_workers=n_workers)
            return out

    def generate_input(self, x):
        x = self.data_augmentation(x)
        return x

    def get_model(self):
        return self.network

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(-1)
    