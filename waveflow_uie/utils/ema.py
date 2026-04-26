"""Exponential Moving Average (EMA) helper with warmup.

EMA decay schedule:
  - Iterations 0 to warmup_iters: decay = warmup_decay (default 0.99)
  - Iterations > warmup_iters:    decay = target_decay (default 0.999)
"""

import copy
from typing import Dict

import torch
import torch.nn as nn


class EMAHelper:
    """Maintains an exponential moving average of model parameters.

    Args:
        model: The model to track.
        target_decay: EMA decay after warmup. Default: 0.999.
        warmup_decay: EMA decay during warmup. Default: 0.99.
        warmup_iters: Number of warmup iterations. Default: 1000.
    """

    def __init__(
        self,
        model: nn.Module,
        target_decay: float = 0.999,
        warmup_decay: float = 0.99,
        warmup_iters: int = 1000,
    ):
        self.target_decay = target_decay
        self.warmup_decay = warmup_decay
        self.warmup_iters = warmup_iters
        self.shadow: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def get_decay(self, step: int) -> float:
        """Get current decay rate based on training step."""
        if step < self.warmup_iters:
            return self.warmup_decay
        return self.target_decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Update shadow parameters with current model parameters."""
        decay = self.get_decay(step)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - decay)

    def apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Swap model parameters with EMA shadow parameters.

        Returns:
            Backup of original parameters (for restore).
        """
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore model parameters from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'target_decay': self.target_decay,
            'warmup_decay': self.warmup_decay,
            'warmup_iters': self.warmup_iters,
        }

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        """Load from checkpoint."""
        self.target_decay = state['target_decay']
        self.warmup_decay = state['warmup_decay']
        self.warmup_iters = state['warmup_iters']
        self.shadow = {k: v.to(device) for k, v in state['shadow'].items()}
