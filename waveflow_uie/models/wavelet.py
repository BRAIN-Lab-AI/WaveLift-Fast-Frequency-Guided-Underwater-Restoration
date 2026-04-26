"""Channel-concatenated Haar Discrete Wavelet Transform (DWT) and Inverse DWT.

Unlike WF-Diff's batch-concatenated wavelet (4N, C, H/2, W/2), this module
produces channel-concatenated output (N, 4C, H/2, W/2) suitable for the
rectified flow model operating in wavelet domain.

Subband ordering: [LL, HL, LH, HH] for each input channel.
For 3-channel RGB input: channels 0-2 = LL(R,G,B), 3-5 = HL, 6-8 = LH, 9-11 = HH.
"""

import torch
import torch.nn as nn
from typing import Tuple


def haar_dwt_2d(x: torch.Tensor) -> torch.Tensor:
    """Compute 1-level 2D Haar DWT with channel concatenation.

    Args:
        x: Input tensor of shape (B, C, H, W). H and W must be even.

    Returns:
        Wavelet coefficients of shape (B, 4*C, H/2, W/2).
        Channel layout: [LL_c0..LL_cC, HL_c0..HL_cC, LH_c0..LH_cC, HH_c0..HH_cC]
    """
    # Split rows into even/odd
    x_even = x[:, :, 0::2, :] / 2.0
    x_odd = x[:, :, 1::2, :] / 2.0

    # Split columns into even/odd
    ee = x_even[:, :, :, 0::2]  # even row, even col
    oe = x_odd[:, :, :, 0::2]   # odd row, even col
    eo = x_even[:, :, :, 1::2]  # even row, odd col
    oo = x_odd[:, :, :, 1::2]   # odd row, odd col

    # Haar wavelet linear combinations
    ll = ee + oe + eo + oo   # Low-Low (approximation)
    hl = -ee - oe + eo + oo  # High-Low
    lh = -ee + oe - eo + oo  # Low-High
    hh = ee - oe - eo + oo   # High-High

    return torch.cat([ll, hl, lh, hh], dim=1)


def haar_idwt_2d(x: torch.Tensor, num_channels: int = 3) -> torch.Tensor:
    """Compute 1-level 2D inverse Haar DWT from channel-concatenated coefficients.

    Args:
        x: Wavelet coefficients of shape (B, 4*C, H/2, W/2).
        num_channels: Number of original image channels C. Default: 3 (RGB).

    Returns:
        Reconstructed tensor of shape (B, C, H*2, W*2).
    """
    c = num_channels
    ll = x[:, 0*c:1*c] / 2.0
    hl = x[:, 1*c:2*c] / 2.0
    lh = x[:, 2*c:3*c] / 2.0
    hh = x[:, 3*c:4*c] / 2.0

    b, _, h_half, w_half = ll.shape
    out = torch.zeros(b, c, h_half * 2, w_half * 2, device=x.device, dtype=x.dtype)

    # Inverse Haar: reconstruct even/odd rows and columns
    out[:, :, 0::2, 0::2] = ll - hl - lh + hh  # even row, even col
    out[:, :, 1::2, 0::2] = ll - hl + lh - hh  # odd row, even col
    out[:, :, 0::2, 1::2] = ll + hl - lh - hh  # even row, odd col
    out[:, :, 1::2, 1::2] = ll + hl + lh + hh  # odd row, odd col

    return out


class HaarDWT2D(nn.Module):
    """1-level 2D Haar DWT module with channel concatenation.

    Input:  (B, C, H, W) where H, W are even
    Output: (B, 4*C, H/2, W/2)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return haar_dwt_2d(x)


class HaarIDWT2D(nn.Module):
    """1-level 2D inverse Haar DWT module.

    Input:  (B, 4*C, H/2, W/2)
    Output: (B, C, H, W)
    """

    def __init__(self, num_channels: int = 3):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return haar_idwt_2d(x, self.num_channels)
