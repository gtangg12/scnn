import functools

import numpy as np
import torch
import pyshtools as sh
from scipy.special import sph_harm

from jaxtyping import Float, Complex


@functools.lru_cache(maxsize=1)
def s2(n: int) -> Float[torch.Tensor, "n n"]:
    """
    :param n: sampling resolution
    """
    return torch.meshgrid(
        torch.linspace(0,     np.pi, n + 1)[:-1],
        torch.linspace(0, 2 * np.pi, n + 1)[:-1], indexing='xy'
    )


@functools.lru_cache(maxsize=1)
def DHaj(bw: int) -> Float[torch.Tensor, "n"]:
    """
    Compute Driscoll-Healy sampling weights

    param bw: bandwidth
    """
    n = 2 * bw
    l = torch.arange(0, bw)
    j = torch.arange(0, n) * np.pi / n
    norm = 1 / (2 * l + 1) * torch.sin((2 * l + 1) * j.reshape(-1, 1))
    return 2 * np.sqrt(2) / n * torch.sin(j) * norm.sum(-1)


@functools.lru_cache(maxsize=1)
def spherical_harmonics(bw: int) -> Complex[torch.Tensor, "l m n n"]:
    """
    param bw: bandwidth
    """
    n = 2 * bw
    phi, theta = s2(n)
    harmonics = torch.zeros(bw, bw, n, n, dtype=torch.complex64)
    for l in range(bw):
        for m in range(0, l + 1):
            harmonics[l, m] = sph_harm(m, l, theta, phi)
    return harmonics


def spherical_transform(
    x             : Float  [torch.Tensor, "b c n n"], 
    a             : Float  [torch.Tensor, "n"],
    harmonics_conj: Complex[torch.Tensor, "l m n n"],
) -> Float[torch.Tensor, "b c l m"]:
    """
    :param x: input signal
    :param a: sampling weights
    :param harmonics: spherical harmonics
    """
    n = x.shape[-1]
    x = x[:, :, None, None, ...] # b c 1 1 n n
    a = a * 2 * np.sqrt(2) * np.pi / n
    return torch.einsum("bclmij,lmij->bclm", (x * a).to(torch.complex64), harmonics_conj)


def spherical_transform_inv(
    fc       : Complex[torch.Tensor, "b c l m"],
    harmonics: Complex[torch.Tensor, "l m n n"],
) -> Float[torch.Tensor, "b c n n"]:
    """
    """
    factor = torch.ones_like(fc) * 2
    factor[:, :, :, 0] = 1
    fc = fc * factor
    real = torch.einsum("bclm,lmij->bcij", torch.real(fc), torch.real(harmonics))
    imag = torch.einsum("bclm,lmij->bcij", torch.imag(fc), torch.imag(harmonics))
    return real - imag


def spherical_conv(
    fc: Complex[torch.Tensor, "b c l m"],
    gc: Complex[torch.Tensor, "d c l 1"],
    wc: Float  [torch.Tensor, "l"],
) -> Complex[torch.Tensor, "b d l m"]:
    """
    """
    wc = wc.unsqueeze(1) # l 1
    fc = fc.unsqueeze(1) # b 1 c l m
    gc = gc.unsqueeze(0) # 1 d c l 1
    return (wc * fc * gc).sum(dim=2)
    

def spectral_pool(fc: Complex[torch.Tensor, "b c l m"]) -> Complex[torch.Tensor, "b c l/2 m/2"]:
    """
    """
    fc[:, :, fc.shape[2] // 2:, :] = 0
    return fc


def magl_pool(fc: Complex[torch.Tensor, "b c l m"]) -> Complex[torch.Tensor, "b c l"]:
    """
    """
    factor = torch.ones_like(fc) * 2
    factor[:, :, :, 0] = 1
    fc = fc * factor
    return torch.linalg.norm(fc, dim=-1)



def sphrot_shtools(f: Float[torch.Tensor, "n n"], x: Float[torch.Tensor, "3"], lmax=None, latcols=True) -> Float[torch.Tensor, "n n"]:
    """ 
    Rotate function on sphere f by Euler angles x (Z-Y-Z convention)
    """
    c = sh.SHGrid.from_array(f.T if latcols else f).expand()
    cr = c.rotate(*x, degrees=False)
    fr = cr.expand(lmax=lmax, grid='DH', extend=False).to_array()
    return fr.T if latcols else fr