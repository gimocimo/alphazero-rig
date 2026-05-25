"""Policy + value network — the learned component of AlphaZero.

Architecture: a small ResNet trunk with two heads.
    Input  →  conv block  →  N residual blocks  →  ┬─→ policy head → logits
                                                   └─→ value head  → scalar in [-1, 1]

Design notes:
    - Channels and depth are configurable. The defaults (32 channels, 3 blocks)
      give ~40k parameters on Connect4 — small enough to train on CPU but
      expressive enough to learn the value function meaningfully.
    - BatchNorm everywhere, the standard AlphaZero choice. BN has a subtle
      caveat: in train mode it requires batch size > 1, so single-state
      inference must use eval() mode (handled by `predict`).
    - The policy head outputs *raw logits*, not probabilities. Masking and
      softmax happen at the inference boundary so the trainer can use stable
      log_softmax-based losses without re-deriving them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..games.base import Game

S = TypeVar("S")


@dataclass
class PVNetConfig:
    """Shape + capacity knobs. Defaults are tuned for Connect4."""

    in_channels: int = 2
    board_h: int = 6
    board_w: int = 7
    action_size: int = 7
    channels: int = 32
    num_blocks: int = 3
    value_hidden: int = 64


class _ResBlock(nn.Module):
    """Standard pre-residual block: conv-BN-relu-conv-BN-add-relu."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PVNet(nn.Module):
    """Two-headed ResNet for AlphaZero-style training.

    Forward signature:
        Input:  (B, C, H, W) — typically (B, 2, 6, 7) for Connect4.
        Output: (policy_logits, value)
                  policy_logits: (B, action_size), raw — no softmax here.
                  value:         (B,) in [-1, 1] (tanh on the final scalar).
    """

    def __init__(self, config: PVNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or PVNetConfig()
        c = self.config

        # Trunk: input conv + residual blocks.
        self.input_conv = nn.Conv2d(c.in_channels, c.channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(c.channels)
        self.blocks = nn.ModuleList(_ResBlock(c.channels) for _ in range(c.num_blocks))

        # Policy head: 1x1 conv to compress channels, then flatten + linear to action_size.
        # The 1x1 acts as a learned per-cell projection before mixing across positions.
        self.policy_conv = nn.Conv2d(c.channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * c.board_h * c.board_w, c.action_size)

        # Value head: 1x1 conv → flatten → MLP → tanh.
        self.value_conv = nn.Conv2d(c.channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(c.board_h * c.board_w, c.value_hidden)
        self.value_fc2 = nn.Linear(c.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = block(x)

        # Policy head.
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head.
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def predict(
    net: PVNet,
    game: Game[S],
    state: S,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, float]:
    """Single-state inference with illegal-move masking.

    Returns:
        probs: (action_size,) float numpy. Illegal moves are zero; legal
            moves are renormalised so the rest sums to 1.
        value: scalar float in [-1, 1], the net's evaluation from the to-move
            player's perspective.

    The mask-then-softmax pattern is what PUCT will consume as the prior.

    If `device` is None we route the input tensor to wherever the net's
    parameters live — avoids the "weights on MPS, inputs on CPU" mismatch
    that bites callers who didn't realise the net had been moved during
    training.
    """
    net.eval()
    if device is None:
        device = next(net.parameters()).device
    x = torch.from_numpy(game.encode(state)).unsqueeze(0).to(device)
    logits, value = net(x)
    logits_np = logits.squeeze(0).cpu().numpy()
    value_f = float(value.item())

    legal = game.legal_actions(state)
    masked = np.where(legal, logits_np, -np.inf)
    # Numerically-stable softmax over the finite entries only.
    if not np.any(np.isfinite(masked)):
        # Pathological — no legal moves. Caller should have caught this.
        raise RuntimeError("predict called on a state with no legal moves")
    masked = masked - np.nanmax(masked[np.isfinite(masked)])
    exp = np.where(legal, np.exp(masked), 0.0)
    probs = exp / exp.sum()
    return probs.astype(np.float32), value_f
