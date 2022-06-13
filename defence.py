import torch.nn as nn


class DefendedNetwork(nn.Module):
    def __init__(self, network, data_augmentation, sample_rate=10, aggregation="mean"):
        nn.Module.__init__(self)
        self.network = network
        self.sample_rate = sample_rate
        self.data_augmentation = data_augmentation
        self.aggregation = aggregation

    def forward(self, x):
        outputs = []
        for n in range(self.sample_rate):
            input = self.generate_input(x)
            out = self.network(input)
            outputs.append(out)
        if self.aggregation == "mean":
            return sum(outputs)/len(outputs)
        return sum(outputs)/len(outputs)

    def generate_input(self, x):
        x = self.data_augmentation(x)
        return x

    def get_model(self):
        return self.network
    