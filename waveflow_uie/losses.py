"""Composite loss for WaveFlow-UIE training.

Components:
  1. L_CFM:          MSE on predicted vs target velocity (rectified flow)
  2. L_freq_weighted: L1 with configurable LL:HF weighting
  3. L_LPIPS:        Perceptual loss on reconstructed RGB (FP32)
  4. L_Lab:          Color consistency in CIELAB space (FP32)
  5. L_physics:      Jaffe-McGlamery auxiliary supervision (FP32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Color space conversion: RGB -> CIELAB (differentiable, FP32)
# ---------------------------------------------------------------------------

def _rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB [0,1] to CIE XYZ. Input/output: (B, 3, H, W)."""
    # Linearize sRGB
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)

    # sRGB to XYZ matrix (D65 illuminant)
    # fmt: off
    m = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], device=rgb.device, dtype=rgb.dtype)
    # fmt: on

    rgb_flat = rgb_linear.permute(0, 2, 3, 1)  # (B, H, W, 3)
    xyz = torch.matmul(rgb_flat, m.T)
    return xyz.permute(0, 3, 1, 2)  # (B, 3, H, W)


def _xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    """Convert CIE XYZ to CIELAB. Input/output: (B, 3, H, W)."""
    # D65 white point
    white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype)
    xyz = xyz / white[None, :, None, None]

    eps = 1e-6
    delta = 6.0 / 29.0
    delta3 = delta ** 3

    mask = xyz > delta3
    f = torch.where(mask, (xyz + eps).pow(1.0 / 3.0), xyz / (3.0 * delta ** 2) + 4.0 / 29.0)

    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])

    return torch.cat([L, a, b], dim=1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to CIELAB. (B, 3, H, W) -> (B, 3, H, W)."""
    return _xyz_to_lab(_rgb_to_xyz(rgb))


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

class CFMLoss(nn.Module):
    """Conditional Flow Matching loss: MSE between predicted and target velocity."""

    def forward(self, v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(v_pred, v_target)


class FreqWeightedLoss(nn.Module):
    """Frequency-weighted L1 loss in wavelet domain.

    Applies different weights to LL (channels 0-2) and HF (channels 3-11) subbands.

    Args:
        hf_weight: Weight multiplier for high-frequency subbands. Default: 2.0.
    """

    def __init__(self, hf_weight: float = 2.0):
        super().__init__()
        self.hf_weight = hf_weight

    def forward(self, v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        err = (v_pred - v_target).abs()
        ll_loss = err[:, :3].mean()
        hf_loss = err[:, 3:].mean()
        return ll_loss + self.hf_weight * hf_loss


class LPIPSLoss(nn.Module):
    """Perceptual loss using LPIPS (AlexNet). Computed in FP32."""

    def __init__(self):
        super().__init__()
        import lpips

        self.lpips_net = lpips.LPIPS(net='alex', verbose=False)
        self.lpips_net.eval()
        for p in self.lpips_net.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: pred, target in [0, 1] range, shape (B, 3, H, W)."""
        with torch.amp.autocast('cuda', enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            # LPIPS expects [-1, 1]
            pred_scaled = pred_fp32 * 2.0 - 1.0
            target_scaled = target_fp32 * 2.0 - 1.0
            return self.lpips_net(pred_scaled, target_scaled).mean()


class LabColorLoss(nn.Module):
    """Color consistency loss in CIELAB space. Computed in FP32."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss on a,b channels of CIELAB. Inputs in [0, 1], shape (B, 3, H, W)."""
        with torch.amp.autocast('cuda', enabled=False):
            pred_lab = rgb_to_lab(pred.float().clamp(0, 1))
            target_lab = rgb_to_lab(target.float().clamp(0, 1))
            # L1 on a and b channels only (color, not luminance)
            return F.l1_loss(pred_lab[:, 1:], target_lab[:, 1:])


class PhysicsLoss(nn.Module):
    """Jaffe-McGlamery auxiliary loss. Computed in FP32.

    Enforces: degraded ≈ clean * t + A * (1 - t)
    """

    def forward(
        self,
        lq: torch.Tensor,
        gt: torch.Tensor,
        t_map: torch.Tensor,
        a_map: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs in [0,1] range. t_map: (B,1,H,W), a_map: (B,3,H,W)."""
        with torch.amp.autocast('cuda', enabled=False):
            lq_fp32 = lq.float()
            gt_fp32 = gt.float()
            t_fp32 = t_map.float()
            a_fp32 = a_map.float()
            reconstructed = gt_fp32 * t_fp32 + a_fp32 * (1.0 - t_fp32)
            return F.mse_loss(lq_fp32, reconstructed)


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class WaveFlowLoss(nn.Module):
    """Composite loss combining all WaveFlow-UIE objectives.

    Args:
        cfm_weight: Weight for flow matching loss. Default: 1.0.
        freq_weight: Weight for frequency-weighted loss. Default: 0.5.
        lpips_weight: Weight for LPIPS loss. Default: 0.1.
        lab_weight: Weight for Lab color loss. Default: 0.05.
        physics_weight: Weight for physics auxiliary loss. Default: 0.1.
        freq_hf_weight: HF multiplier within freq loss. Default: 2.0.
        physics_enabled: Whether to compute physics loss. Default: True.
    """

    def __init__(
        self,
        cfm_weight: float = 1.0,
        freq_weight: float = 0.5,
        lpips_weight: float = 0.1,
        lab_weight: float = 0.05,
        physics_weight: float = 0.1,
        freq_hf_weight: float = 2.0,
        physics_enabled: bool = True,
    ):
        super().__init__()
        self.cfm_weight = cfm_weight
        self.freq_weight = freq_weight
        self.lpips_weight = lpips_weight
        self.lab_weight = lab_weight
        self.physics_weight = physics_weight
        self.physics_enabled = physics_enabled

        self.cfm_loss = CFMLoss()
        self.freq_loss = FreqWeightedLoss(hf_weight=freq_hf_weight)
        self.lpips_loss = LPIPSLoss() if lpips_weight > 0 else None
        self.lab_loss = LabColorLoss()
        self.physics_loss = PhysicsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        lq: torch.Tensor,
        gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            outputs: Dictionary from WaveFlowUIE.training_step().
            lq: Degraded RGB (B, 3, H, W).
            gt: Clean RGB (B, 3, H, W).

        Returns:
            Dictionary with 'total' and individual loss components.
        """
        losses = {}

        # 1. CFM loss
        losses['cfm'] = self.cfm_loss(outputs['v_pred'], outputs['v_target'])

        # 2. Frequency-weighted loss
        losses['freq'] = self.freq_loss(outputs['v_pred'], outputs['v_target'])

        # 3. LPIPS loss
        if self.lpips_loss is not None and self.lpips_weight > 0:
            losses['lpips'] = self.lpips_loss(outputs['pred_rgb'], gt)
        else:
            losses['lpips'] = torch.tensor(0.0, device=gt.device)

        # 4. Lab color loss
        losses['lab'] = self.lab_loss(outputs['pred_rgb'], gt)

        # 5. Physics loss
        if self.physics_enabled and self.physics_weight > 0:
            losses['physics'] = self.physics_loss(lq, gt, outputs['t_map'], outputs['a_map'])
        else:
            losses['physics'] = torch.tensor(0.0, device=lq.device)

        # Total
        losses['total'] = (
            self.cfm_weight * losses['cfm']
            + self.freq_weight * losses['freq']
            + self.lpips_weight * losses['lpips']
            + self.lab_weight * losses['lab']
            + self.physics_weight * losses['physics']
        )

        return losses
