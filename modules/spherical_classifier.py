import torch
from torch import nn
from jaxtyping import Float

from modules.spherical_nn import *


class SphericalResidualBlock(nn.Module):
    """
    """
    def __init__(self, in_channels: int, out_channels: int, bandwidth: int):
        super().__init__()
        self.conv = SphericalConv(in_channels, out_channels, bandwidth)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = x + self.conv(x)
        x = self.relu(x)
        return x


class SphericalClassifer(nn.Module):
    """
    """
    def __init__(self, bandwidth: int, num_classes: int):
        """
        """
        super().__init__()
        self.scnn_upper1 = nn.Sequential(*[
            SphericalConv(1, 32, bandwidth),
            nn.PReLU(),
            SphericalResidualBlock(32, 32, bandwidth),
            SphericalResidualBlock(32, 32, bandwidth),
            nn.MaxPool2d(2),
        ])
        bandwidth = bandwidth // 2
        self.scnn_upper2 = nn.Sequential(*[
            SphericalResidualBlock(32, 32, bandwidth),
        ])
        self.scnn_lower1 = nn.Sequential(*[
            SphericalConv(32, 64, bandwidth),
            nn.PReLU(),
            SphericalResidualBlock(64, 64, bandwidth),
            SphericalResidualBlock(64, 64, bandwidth),
            SphericalResidualBlock(64, 64, bandwidth),
            nn.MaxPool2d(2),
        ])
        bandwidth = bandwidth // 2
        self.scnn_lower2 = nn.Sequential(*[
            SphericalConv(64, 128, bandwidth),
            nn.PReLU(),
            SphericalResidualBlock(128, 128, bandwidth),
            SphericalResidualBlock(128, 128, bandwidth),
            SphericalResidualBlock(128, 128, bandwidth),
            SphericalResidualBlock(128, 128, bandwidth),
        ])
        self.classifer = nn.Sequential(*[
            MagLPool(bandwidth),
            nn.Flatten(),
            nn.PReLU(),
            nn.Linear(128 * bandwidth, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes),
        ])

    def forward(self, x: Float[torch.Tensor, "b c n n"], return_features=False) -> Float[torch.Tensor, "b num_classes"]:
        """
        """
        x = self.scnn_upper1(x)
        x = self.scnn_upper2(x)
        if return_features:
            return x
        x = self.scnn_lower1(x)
        x = self.scnn_lower2(x)
        x = self.classifer(x)
        return x
    

if __name__ == '__main__':
    model = SphericalClassifer(32, 40).to('cuda')
    x = torch.randn(10, 1, 64, 64).to('cuda')
    y = model(x)
    print(y.shape)