"""WaveFlow-UIE: Full pipeline wrapping DWT, physics prior, rectified flow, and IDWT.

This module orchestrates the complete forward pass for both training and inference:
  Training:  degraded_rgb -> DWT -> physics -> flow matching -> losses
  Inference: degraded_rgb -> DWT -> physics -> ODE solve (Euler/midpoint) -> IDWT -> enhanced_rgb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .wavelet import HaarDWT2D, HaarIDWT2D
from .physics_prior import PhysicsPriorNet
from .velocity_unet import VelocityUNet


class WaveFlowUIE(nn.Module):
    """Wavelet-domain rectified flow model for underwater image enhancement.

    Args:
        unet_kwargs: Keyword arguments for VelocityUNet.
        physics_kwargs: Keyword arguments for PhysicsPriorNet.
    """

    def __init__(
        self,
        unet_kwargs: Optional[dict] = None,
        physics_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D(num_channels=3)
        self.physics_prior = PhysicsPriorNet(**(physics_kwargs or {}))
        self.velocity_unet = VelocityUNet(**(unet_kwargs or {}))

    def training_step(
        self,
        lq: torch.Tensor,
        gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute rectified flow training outputs.

        Args:
            lq: Degraded RGB, shape (B, 3, H, W).
            gt: Clean RGB, shape (B, 3, H, W).

        Returns:
            Dictionary with keys:
              - v_pred: predicted velocity (B, 12, H/2, W/2)
              - v_target: target velocity (B, 12, H/2, W/2)
              - w_pred_clean: predicted clean wavelets = w_lq + v_pred (B, 12, H/2, W/2)
              - pred_rgb: IDWT(w_pred_clean) (B, 3, H, W)
              - physics_cond: physics conditioning (B, 4, H/2, W/2)
              - t_map: transmission map at full res (B, 1, H, W)
              - a_map: ambient light at full res (B, 3, H, W)
        """
        b = lq.shape[0]

        # Wavelet decomposition
        w_lq = self.dwt(lq)    # (B, 12, H/2, W/2)
        w_gt = self.dwt(gt)    # (B, 12, H/2, W/2)

        # Physics prior
        target_size = (w_lq.shape[2], w_lq.shape[3])
        physics_cond, t_map, a_map = self.physics_prior(lq, target_size=target_size)

        # Rectified flow: sample random time t ~ U[0, 1]
        t = torch.rand(b, device=lq.device, dtype=lq.dtype)
        t_expanded = t.view(b, 1, 1, 1)

        # Linear interpolation: x_t = (1 - t) * w_lq + t * w_gt
        x_t = (1.0 - t_expanded) * w_lq + t_expanded * w_gt

        # Target velocity: constant for rectified flow
        v_target = w_gt - w_lq

        # Predict velocity
        unet_input = torch.cat([x_t, w_lq, physics_cond], dim=1)  # (B, 28, H/2, W/2)
        v_pred = self.velocity_unet(unet_input, t)

        # Reconstruct predicted clean wavelets (for perceptual losses)
        w_pred_clean = w_lq + v_pred
        pred_rgb = self.idwt(w_pred_clean)

        return {
            'v_pred': v_pred,
            'v_target': v_target,
            'w_pred_clean': w_pred_clean,
            'pred_rgb': pred_rgb,
            'physics_cond': physics_cond,
            't_map': t_map,
            'a_map': a_map,
        }

    @torch.no_grad()
    def sample(
        self,
        lq: torch.Tensor,
        num_steps: int = 5,
        solver: str = 'euler',
    ) -> torch.Tensor:
        """Inference: enhance degraded image via ODE integration.

        Args:
            lq: Degraded RGB, shape (B, 3, H, W). H, W must be even.
            num_steps: Number of ODE steps. Default: 5.
            solver: 'euler' or 'midpoint'. Default: 'euler'.

        Returns:
            Enhanced RGB, shape (B, 3, H, W), clamped to [0, 1].
        """
        w_lq = self.dwt(lq)
        target_size = (w_lq.shape[2], w_lq.shape[3])
        physics_cond, _, _ = self.physics_prior(lq, target_size=target_size)

        w = w_lq.clone()
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((lq.shape[0],), t_val, device=lq.device, dtype=lq.dtype)

            if solver == 'euler':
                unet_input = torch.cat([w, w_lq, physics_cond], dim=1)
                v = self.velocity_unet(unet_input, t)
                w = w + v * dt

            elif solver == 'midpoint':
                # First evaluation
                unet_input = torch.cat([w, w_lq, physics_cond], dim=1)
                v1 = self.velocity_unet(unet_input, t)
                w_mid = w + v1 * (dt / 2.0)

                # Second evaluation at midpoint
                t_mid = torch.full((lq.shape[0],), t_val + dt / 2.0, device=lq.device, dtype=lq.dtype)
                unet_input_mid = torch.cat([w_mid, w_lq, physics_cond], dim=1)
                v2 = self.velocity_unet(unet_input_mid, t_mid)
                w = w + v2 * dt

            else:
                raise ValueError(f"Unknown solver: {solver}. Use 'euler' or 'midpoint'.")

        enhanced = self.idwt(w)
        return enhanced.clamp(0.0, 1.0)
