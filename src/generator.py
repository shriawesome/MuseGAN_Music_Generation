import torch
from utils import config 

# Variable to read the constant values
conf = config.Config

class GeneratorBlock(torch.nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_filter, stride):
        super().__init__()
        self.convtrans = torch.nn.ConvTranspose3d(inp_dim, out_dim, kernel_filter, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)

    """
    Feed forward function

    Parameters
    -----------
    x : input data for the model

    Returns
    -------
    Value from Relu activation
    """
    def forward(self, x):
        x = self.convtrans(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)


class Generator(torch.nn.Module):
    """
    A Convolutional Neural Network(CNN) based generator. It takes latent vector
    as an input and outputs pseudo samples.
    """
    def __init__(self):
        super().__init__()
        self.convtrans0 = GeneratorBlock(conf.LATENT_DIM, 256, (4, 1, 1),(4, 1, 1))
        self.convtrans1 = GeneratorBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.convtrans2 = GeneratorBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.convtrans3 = GeneratorBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.convtrans4 = torch.nn.ModuleList([GeneratorBlock(32, 16, (1, 4, 1), (1, 4, 1)) for _ in range(conf.N_TRACKS)])
        self.convtrans5 = torch.nn.ModuleList([GeneratorBlock(16, 1, (1, 1, 12), (1, 1, 12)) for _ in range(conf.N_TRACKS)])


    """
    Feed forward method

    Parameters
    -----------
    x : input data for the model

    Returns
    -------
    Value from Relu activation
    """
    def forward(self, x):
        x = x.view(-1, conf.LATENT_DIM, 1, 1, 1)
        x = self.convtrans0(x)
        x = self.convtrans1(x)
        x = self.convtrans2(x)
        x = self.convtrans3(x)
        x = [convtrans(x) for convtrans in self.convtrans4]
        x = torch.cat([convtrans(x_in) for x_in, convtrans in zip(x, self.convtrans5)],1)
        x = x.view(-1, conf.N_TRACKS, conf.N_MEASURES * conf.MEASURE_RESOLUTION, conf.N_PITCHES)
        return x