"""Unit tests for wavelet transform."""

import torch
import pytest
from waveflow_uie.models.wavelet import haar_dwt_2d, haar_idwt_2d, HaarDWT2D, HaarIDWT2D


def test_dwt_output_shape():
    x = torch.randn(2, 3, 256, 256)
    w = haar_dwt_2d(x)
    assert w.shape == (2, 12, 128, 128)


def test_idwt_output_shape():
    w = torch.randn(2, 12, 128, 128)
    x = haar_idwt_2d(w, num_channels=3)
    assert x.shape == (2, 3, 256, 256)


def test_perfect_reconstruction():
    x = torch.randn(4, 3, 128, 128)
    w = haar_dwt_2d(x)
    x_recon = haar_idwt_2d(w, num_channels=3)
    assert (x - x_recon).abs().max() < 1e-6


def test_constant_image_subbands():
    """Constant image should have zero high-frequency subbands."""
    x = torch.ones(1, 3, 64, 64) * 0.5
    w = haar_dwt_2d(x)
    hf = w[:, 3:]  # HL, LH, HH
    assert hf.abs().max() < 1e-6


def test_various_shapes():
    for b, c, h, w_dim in [(1, 1, 32, 32), (2, 6, 64, 64), (1, 3, 16, 16)]:
        x = torch.randn(b, c, h, w_dim)
        wt = haar_dwt_2d(x)
        assert wt.shape == (b, 4 * c, h // 2, w_dim // 2)
        x_r = haar_idwt_2d(wt, num_channels=c)
        assert (x - x_r).abs().max() < 1e-6


def test_module_interface():
    dwt = HaarDWT2D()
    idwt = HaarIDWT2D(num_channels=3)
    x = torch.randn(2, 3, 64, 64)
    w = dwt(x)
    x_r = idwt(w)
    assert (x - x_r).abs().max() < 1e-6


def test_gradient_flow():
    """DWT/IDWT should allow gradient flow."""
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    w = haar_dwt_2d(x)
    loss = w.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
