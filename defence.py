import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributions as distributions



class DataAugmenter(nn.Module):
    def __init__(self, max_shift=0.1, p=0.5):
        nn.Module.__init__(self)
        self.jitter = transforms.ColorJitter()
        self.shift = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=0, translate=(0, max_shift)),
            transforms.ToTensor()
            ]) 
        self.p=p
        return

    def forward(self, x):
        val = torch.rand(1)
        if val < self.p:
            if val < self.p:
                return self.jitter(x)
            else:
                #x = self.shift(x)
                return x
        else:
            return x
        

class DefendedNetwork(nn.Module):
    def __init__(self, network, epsilon=0.5, sample_rate=10):
        nn.Module.__init__(self)
        self.network = network
        self.epsilon = epsilon
        self.sample_rate = sample_rate
        self.distribution = distributions.uniform.Uniform(0, 1)
        self.da = DataAugmenter()

    def forward(self, x):
        outputs = []
        for n in range(self.sample_rate):
            input = self.generate_input(x)
            out = self.network(input)
            outputs.append(out)
        return sum(outputs)/len(outputs)

    def generate_input(self, x):
        x = self.da(x)
        noise = (self.distribution.sample(x.size()) * self.epsilon).to(device="cuda")
        input = x + noise
        return input
