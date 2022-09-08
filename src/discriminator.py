import torch
from utils import config 

# Variable to read the constant values
conf = config.Config

class LayerNorm(torch.nn.Module):
    """
    Implementation of Layer Normalization that does not require size
    information
    """
    def __init__(self, n_features, eps = 1e-5, affine = True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.alpha = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    
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
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean)/(std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y =self.alpha.view(*shape) * y + self.beta.view(*shape)
        
        return y


class DiscriminatorBlock(torch.nn.Module):
    """
    Basic building block to be used for creating a discriminator network.
    """
    def __init__(self, inp_dim, out_dim, kernel_filter, stride):
        self.convtrans = torch.nn.Conv3d(inp_dim, out_dim, kernel_filter, stride)
        self.layer_norm = LayerNorm(out_dim)


    """
    Feed forward method

    Parameters
    -----------
    x : input data for the model

    Returns
    -------
    Value from leaky relu activation
    """
    def forward(self, x):
        x = self.convtrans(x)
        x = self.layer_norm(x)
        return torch.nn.functional.leaky_relu(x)


class Discriminator(torch.nn.Module):
    """
    A Convolutional Neural Network(CNN) based discriminator. The discriminator takes
    as input either a real sample or a fake sample and outputs a scalar indicating
    its authenticity.
    """
    def __init(self):
        super().__init__()
        self.conv0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(conf.N_TRACKS)
        ])
        self.conv1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 16, (1, 4, 1), (1, 4, 1)) for _ in range(conf.N_TRACKS)
        ])
        self.conv2 = DiscriminatorBlock(16 * 5, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = DiscriminatorBlock(64, 64, (1, 1, 4), (1, 1, 4))
        self.conv4 = DiscriminatorBlock(64, 128, (1, 4, 1), (1, 4, 1))
        self.conv5 = DiscriminatorBlock(128, 128, (2, 1, 1), (1, 1, 1))
        self.conv6 = DiscriminatorBlock(128, 256, (3, 1, 1), (3, 1, 1))
        self.dense = torch.nn.Linear(256, 1)


    """
    Feed forward method

    Parameters
    -----------
    x : input data for the model

    Returns
    -------
    Scalar value
    """
    def forward(self, x):
        x = x.view(-1, conf.N_TRACKS, conf.N_MEASURES, conf.MEASURE_RESOLUTION, conf.N_PITCHES)
        x = [conv(x[:, [i]]) for i, conv in enumerate(self.conv0)]
        x = torch.cat([conv(x_) for x_, conv in zip(x, self.conv1)], 1)
        x = self.conv2(x)
        x = self.conv3(x)          
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 256)
        x = self.dense(x)
        return x