import torch
import torch.nn as nn
from jaxtyping import Float

from modules.spherical_functional import *


class SpectralPool(nn.Module):
    """
    """
    def forward(self, x: Complex[torch.Tensor, "b c l m"]) -> Complex[torch.Tensor, "b c l m"]:
        """
        :param x: input coefficients
        """
        return spectral_pool(x)
    

class MagLPool(nn.Module):
    """
    """
    def __init__(self, bandwidth: int):
        """
        """
        super().__init__()
        self.a             :Float  [torch.Tensor, "n"]       = nn.Parameter(DHaj(bandwidth)               , requires_grad=False)
        self.harmonics     :Complex[torch.Tensor, "l m n n"] = nn.Parameter(spherical_harmonics(bandwidth), requires_grad=False)
        self.harmonics_conj:Complex[torch.Tensor, "l m n n"] = nn.Parameter(torch.conj(self.harmonics)    , requires_grad=False)
    
    def forward(self, x: Complex[torch.Tensor, "b c n n"]) -> Complex[torch.Tensor, "b c l"]:
        """
        :param x: input function on S2
        """
        c = spherical_transform(x, self.a, self.harmonics_conj)
        x = magl_pool(c).float()
        return x
    

class SphericalConv(nn.Module):
    """
    """
    def __init__(self, in_channels: int, out_channels: int, bandwidth: int, bias=True, spectral_pool=False):
        """
        bandwidth: bandwidth of the spherical harmonics
        """
        super().__init__()
        self.a             :Float  [torch.Tensor, "n"]       = nn.Parameter(DHaj(bandwidth)               , requires_grad=False)
        self.harmonics     :Complex[torch.Tensor, "l m n n"] = nn.Parameter(spherical_harmonics(bandwidth), requires_grad=False)
        self.harmonics_conj:Complex[torch.Tensor, "l m n n"] = nn.Parameter(torch.conj(self.harmonics)    , requires_grad=False)

        std = 1 / (np.pi * np.sqrt((bandwidth) * (in_channels)))
        self.weights: Complex[torch.Tensor, "d c l 1"] = nn.Parameter(torch.randn(out_channels, in_channels, bandwidth, 1, dtype=torch.complex64) * std)
        self.bias   : Float  [torch.Tensor, "d"]       = nn.Parameter(torch.zeros(out_channels, 1, 1)) if bias else 0
        self.mult   : Float  [torch.Tensor, "l"]       = nn.Parameter(2 * np.pi / torch.sqrt(4 * np.pi / torch.arange(bandwidth)), requires_grad=False)

        self.pool = SpectralPool() if spectral_pool else nn.Identity()

    def forward(self, x: Float[torch.Tensor, "b c n n"]) -> Float[torch.Tensor, "b d n n"]:
        """
        :param x: input function on S2
        """
        c = spherical_transform(x, self.a, self.harmonics_conj)
        c = spherical_conv(c, self.weights, self.mult)
        c = self.pool(c)
        x = spherical_transform_inv(c, self.harmonics)
        x = x + self.bias
        return x