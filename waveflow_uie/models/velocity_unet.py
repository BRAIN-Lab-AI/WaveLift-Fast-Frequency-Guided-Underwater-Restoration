"""Conditional Velocity UNet for rectified flow in wavelet domain.

Predicts velocity v_theta(x_t, t, condition) where:
  - x_t: noisy wavelet state at time t, shape (B, 12, H, W)
  - t: flow time in [0, 1]
  - condition: concatenation of degraded wavelets (12ch) + physics prior (4ch)

Architecture: standard UNet with AdaGN time conditioning, multi-head self-attention
at low spatial resolutions (16x16 and 8x8).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t: (B,) float tensor in [0, 1]. Returns: (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeMLPEmbedding(nn.Module):
    """Sinusoidal embedding followed by 2-layer MLP."""

    def __init__(self, sinusoidal_dim: int = 64, time_emb_dim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# ---------------------------------------------------------------------------
# AdaGN ResBlock
# ---------------------------------------------------------------------------

class AdaGNResBlock(nn.Module):
    """Residual block with Adaptive Group Normalization for time conditioning.

    AdaGN: GroupNorm(h) * (1 + scale) + shift,
    where (scale, shift) = Linear(time_emb).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = 256,
        num_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Args: x: (B,C,H,W), t_emb: (B, time_emb_dim)."""
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # AdaGN: condition on time
        t_proj = self.time_proj(self.act(t_emb))[:, :, None, None]  # (B, 2*C, 1, 1)
        scale, shift = t_proj.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_conv(x)


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, heads, head_dim, H*W)

        # Transpose for attention: (B, heads, H*W, head_dim)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, H*W, head_dim)

        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.proj(out) + residual


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


# ---------------------------------------------------------------------------
# Velocity UNet
# ---------------------------------------------------------------------------

class VelocityUNet(nn.Module):
    """Conditional UNet predicting velocity in wavelet domain.

    Args:
        in_channels: Input channels (flow state + condition + physics). Default: 28.
        out_channels: Output channels (predicted velocity). Default: 12.
        base_channels: Base channel count. Default: 64.
        channel_mults: Channel multipliers per level. Default: (1, 2, 4, 8).
        num_res_blocks: ResBlocks per level. Default: 2.
        attention_resolutions: Spatial resolutions where self-attention is applied.
            Default: (16, 8) — attention at 16x16 and 8x8.
        dropout: Dropout rate. Default: 0.1.
        time_emb_dim: Time embedding dimension. Default: 256.
        num_groups: GroupNorm groups. Default: 32.
    """

    def __init__(
        self,
        in_channels: int = 28,
        out_channels: int = 12,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.1,
        time_emb_dim: int = 256,
        num_groups: int = 32,
    ):
        super().__init__()
        self.attention_resolutions = set(attention_resolutions)
        self.num_res_blocks = num_res_blocks
        self.channel_mults = channel_mults
        self.num_levels = len(channel_mults)

        # Time embedding
        self.time_embed = TimeMLPEmbedding(sinusoidal_dim=64, time_emb_dim=time_emb_dim)

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build encoder levels
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()

        channels = [base_channels]
        ch = base_channels
        current_res = 128  # starting spatial resolution (from 256x256 input after DWT)

        for level, mult in enumerate(channel_mults):
            ch_out = base_channels * mult

            for _ in range(num_res_blocks):
                layers = nn.ModuleList([
                    AdaGNResBlock(ch, ch_out, time_emb_dim, num_groups, dropout)
                ])
                if current_res in self.attention_resolutions:
                    layers.append(MultiHeadSelfAttention(ch_out, num_heads=max(1, ch_out // 64)))
                self.encoder_blocks.append(layers)
                ch = ch_out
                channels.append(ch)

            if level < len(channel_mults) - 1:
                self.encoder_downsamples.append(Downsample(ch))
                channels.append(ch)
                current_res //= 2
            else:
                self.encoder_downsamples.append(None)

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            AdaGNResBlock(ch, ch, time_emb_dim, num_groups, dropout),
            MultiHeadSelfAttention(ch, num_heads=max(1, ch // 64)),
            AdaGNResBlock(ch, ch, time_emb_dim, num_groups, dropout),
        ])

        # Build decoder levels
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            ch_out = base_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = nn.ModuleList([
                    AdaGNResBlock(ch + skip_ch, ch_out, time_emb_dim, num_groups, dropout)
                ])
                if current_res in self.attention_resolutions:
                    layers.append(MultiHeadSelfAttention(ch_out, num_heads=max(1, ch_out // 64)))
                self.decoder_blocks.append(layers)
                ch = ch_out

            if level > 0:
                self.decoder_upsamples.append(Upsample(ch))
                current_res *= 2
            else:
                self.decoder_upsamples.append(None)

        # Output
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Concatenated input (flow_state, condition, physics), shape (B, 28, H, W).
            t: Flow time, shape (B,) in [0, 1].

        Returns:
            Predicted velocity, shape (B, 12, H, W).
        """
        t_emb = self.time_embed(t)
        h = self.input_conv(x)

        # Encoder
        skips = [h]
        block_idx = 0
        for level_idx in range(self.num_levels):
            for _ in range(self.num_res_blocks):
                block = self.encoder_blocks[block_idx]
                h = block[0](h, t_emb)
                if len(block) > 1:
                    h = block[1](h)
                skips.append(h)
                block_idx += 1

            ds = self.encoder_downsamples[level_idx]
            if ds is not None:
                h = ds(h)
                skips.append(h)

        # Bottleneck
        h = self.bottleneck[0](h, t_emb)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, t_emb)

        # Decoder
        block_idx = 0
        for level_idx in range(len(self.decoder_upsamples)):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                # Pad h if spatial dims are smaller than skip (odd-size rounding)
                if h.shape[2:] != skip.shape[2:]:
                    diff_h = skip.shape[2] - h.shape[2]
                    diff_w = skip.shape[3] - h.shape[3]
                    h = F.pad(h, (0, diff_w, 0, diff_h))
                h = torch.cat([h, skip], dim=1)
                block = self.decoder_blocks[block_idx]
                h = block[0](h, t_emb)
                if len(block) > 1:
                    h = block[1](h)
                block_idx += 1

            up = self.decoder_upsamples[level_idx]
            if up is not None:
                h = up(h)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)
