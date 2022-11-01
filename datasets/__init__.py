import torch
data_root = "data/"


def sample_torch_dataset(dset, batch_size=32, shuffle=False):
    f, h = dset[0]
    X_shape = f.shape
    # We assume y is a scalar output
    batch_shape = (batch_size, ) + X_shape
    y_shape = (batch_size, )
    device = f.device
    X = torch.zeros(size=batch_shape).to(device)
    y = torch.zeros(size=y_shape).to(device)
    if shuffle:
        idx = torch.randint(low=0, high=len(dset), size=(batch_size, ))
    else:
        idx = range(batch_size)
    for i, id in enumerate(idx):
        X[i], y[i] = dset[id]
    return X, y