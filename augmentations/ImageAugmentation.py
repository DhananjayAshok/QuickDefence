import torch
from imgaug import augmenters as iaa
from .DataAugmentation import DataAugmentation


class ImageAugmentation(DataAugmentation):
    def __init__(self, sequence):
        """

        :param sequence: When seq(inp=inp) is called returns processed augmented input
        """
        DataAugmentation.__init__(self, sequence=sequence)

    def __call__(self, inp):
        """
        Currently we assume all augmentations are imgaug augmenters which accept input in shape BS, H, W, C

        :param inp: input to the neural network we seek to protect
        :return: augmented input x
        """
        device = None
        if isinstance(inp, torch.Tensor):
            device = inp.device
            if len(inp.shape) == 4:
                bs, C, H, W = inp.shape
                inp = inp.reshape(bs, H, W, C)
            else:
                C, H, W = inp.shape
                inp = inp.reshape(H, W, C)
            inp = inp.cpu().detach().numpy()
        if len(inp.shape) == 4:
            inp = self.sequence(images=inp)
        else:
            inp = self.sequence(image=inp)
        if device is not None:
            inp = torch.Tensor(inp).to(device)
        if isinstance(inp, torch.Tensor):
            if len(inp.shape) == 4:
                bs, H, W, C = inp.shape
                inp = inp.reshape(bs, C, H, W)
            else:
                H, W, C = inp.shape
                inp = inp.reshape(C, H, W)
        return inp


class Noise(ImageAugmentation):
    def __init__(self, dist="gaussian", dist_params={}):

        if dist == "gaussian":
            seq = iaa.AdditiveGaussianNoise(loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 0.01),
                                            per_channel=True)
        elif dist == "laplace":
            seq = iaa.AdditiveLaplaceNoise(loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 0.01),
                                           per_channel=True)
        elif dist == "poisson":
            seq = iaa.AdditivePoissonNoise(lam=dist_params.get('lam', 1), scale=dist_params.get('scale', 0.01),
                                           per_channel=True)
        ImageAugmentation.__init__(self=self, sequence=seq)


class Affine(ImageAugmentation):
    def __init__(self, scale_x=(0.995, 1.005), scale_y=(0.995, 1.005), trans_x=(0, 0), trans_y=(0, 0), rotate_l=-1
                 , rotate_u=1):
        seq = iaa.Affine(scale={"x": scale_x, "y": scale_y}, translate_percent={"x": trans_x, "y": trans_y},
                         rotate=(rotate_l, rotate_u))
        ImageAugmentation.__init__(self=self, sequence=seq)


class Saturation(ImageAugmentation):
    def __init__(self, low=-5, high=5):
        inseq = iaa.AddToSaturation((low, high))
        ImageAugmentation.__init__(self, sequence=inseq)


class Hue(ImageAugmentation):
    def __init__(self, low=-5, high=5):
        inseq = iaa.AddToHue((low, high))
        ImageAugmentation.__init__(self, sequence=inseq)


class Brightness(ImageAugmentation):
    def __init__(self, low=-3, high=3):
        inseq = iaa.AddToHue((low, high))
        ImageAugmentation.__init__(self, sequence=inseq)