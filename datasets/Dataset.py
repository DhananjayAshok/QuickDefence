from torch.utils.data import Dataset as torchDset

data_root = "data/"


class Dataset(torchDset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert len(X) == len(y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)
