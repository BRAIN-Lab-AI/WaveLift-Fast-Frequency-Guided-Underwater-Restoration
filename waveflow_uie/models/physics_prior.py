"""Physics Prior Branch for underwater image formation model.

Estimates transmission map t(x) and ambient light A from degraded input
based on the Jaffe-McGlamery model: Y = X * t + A * (1 - t).

Outputs 4 conditioning channels at half spatial resolution:
  - 1 channel: transmission map (sigmoid, [0,1])
  - 3 channels: ambient light per RGB channel (sigmoid, [0,1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PhysicsPriorNet(nn.Module):
    """Lightweight CNN estimating underwater imaging parameters.

    Args:
        in_channels: Input image channels. Default: 3.
        hidden_channels: Hidden layer channels. Default: 32.
        enabled: If False, forward returns zeros (for ablation). Default: True.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = enabled

        # Shared backbone: 3 conv layers
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Transmission head: spatial map
        self.transmission_head = nn.Conv2d(hidden_channels, 1, 1)

        # Ambient light head: global estimate per RGB channel
        self.ambient_pool = nn.AdaptiveAvgPool2d(1)
        self.ambient_fc = nn.Linear(hidden_channels, 3)

    def forward(
        self, x: torch.Tensor, target_size: Tuple[int, int] = (128, 128)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate physics priors from degraded input.

        Args:
            x: Degraded RGB image, shape (B, 3, H, W).
            target_size: Spatial size to downsample outputs to. Default: (128, 128).

        Returns:
            physics_cond: Combined conditioning, shape (B, 4, target_H, target_W).
            t_map: Full-resolution transmission map, shape (B, 1, H, W).
            a_map: Full-resolution ambient light, shape (B, 3, H, W).
        """
        b = x.shape[0]

        if not self.enabled:
            t_map = torch.zeros(b, 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            a_map = torch.zeros(b, 3, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            physics_cond = torch.zeros(b, 4, target_size[0], target_size[1], device=x.device, dtype=x.dtype)
            return physics_cond, t_map, a_map

        features = self.backbone(x)

        # Transmission: per-pixel, sigmoid to [0, 1]
        t_map = torch.sigmoid(self.transmission_head(features))  # (B, 1, H, W)

        # Ambient light: global per-channel, sigmoid to [0, 1]
        pooled = self.ambient_pool(features).view(b, -1)  # (B, hidden)
        a_vec = torch.sigmoid(self.ambient_fc(pooled))  # (B, 3)
        a_map = a_vec[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])  # (B, 3, H, W)

        # Downsample and concatenate for UNet conditioning
        t_down = F.interpolate(t_map, size=target_size, mode='bilinear', align_corners=False)
        a_down = F.interpolate(a_map, size=target_size, mode='bilinear', align_corners=False)
        physics_cond = torch.cat([t_down, a_down], dim=1)  # (B, 4, H/2, W/2)

        return physics_cond, t_map, a_map
