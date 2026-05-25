"""Trainer — turns replay-buffer samples into network updates.

One training step:
    1. Sample a minibatch from the replay buffer.
    2. Forward pass through the net → (policy_logits, value).
    3. Compute the two-headed loss:
         L_policy = -mean(sum(target_policy * log_softmax(policy_logits)))
         L_value  = mse(value, target_value)
         L_total  = L_policy + L_value      (plus L2 from optimizer's weight_decay)
    4. Backward + optionally clip gradient norm + optimizer step.

Notes on the loss shape:
    - `L_policy` is a *soft* cross-entropy: the target is a distribution
      (MCTS visit fractions), not a one-hot label. The closed-form
      log_softmax-based formulation is more numerically stable than
      softmax-then-log applied separately, which is why PVNet returns raw
      logits rather than probabilities.
    - The two losses are summed with equal weight. The original AlphaZero
      paper weights them equally too (the value loss has an implicit factor
      from MSE's scale matching CE in the regime we operate). Some later
      implementations weight L_value lower (e.g. 0.5) — we can revisit if
      training is unstable.

Device handling: auto-detect MPS (Apple Silicon) → CUDA → CPU. Override via
TrainerConfig.device if needed (e.g. for CPU-only benchmarks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..nets.pvnet import PVNet, PVNetConfig
from .replay_buffer import ReplayBuffer


def _auto_detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularisation on net parameters
    batch_size: int = 128
    grad_clip: float | None = 1.0  # max L2 norm of gradients; None disables
    device: str | None = None  # auto-detect if None


class Trainer:
    """Owns network + optimizer + step counter. One method matters: `train_step`."""

    def __init__(self, net: PVNet, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.device = self.config.device or _auto_detect_device()
        self.net = net.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.step_count = 0

    def train_step(self, buffer: ReplayBuffer) -> dict[str, float]:
        """Sample a batch, take one gradient step, return loss metrics."""
        states_np, policies_np, values_np = buffer.sample(self.config.batch_size)
        states = torch.from_numpy(states_np).to(self.device)
        target_policies = torch.from_numpy(policies_np).to(self.device)
        target_values = torch.from_numpy(values_np).to(self.device)

        self.net.train()
        self.optimizer.zero_grad()
        logits, predicted_values = self.net(states)

        log_probs = F.log_softmax(logits, dim=-1)
        loss_policy = -(target_policies * log_probs).sum(dim=-1).mean()
        loss_value = F.mse_loss(predicted_values, target_values)
        loss = loss_policy + loss_value

        loss.backward()
        grad_norm = None
        if self.config.grad_clip is not None:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.config.grad_clip
            )
            grad_norm = float(grad_norm_t)
        self.optimizer.step()
        self.step_count += 1

        return {
            "loss": float(loss.item()),
            "loss_policy": float(loss_policy.item()),
            "loss_value": float(loss_value.item()),
            "grad_norm": grad_norm if grad_norm is not None else float("nan"),
            "step": self.step_count,
        }

    def train(self, buffer: ReplayBuffer, steps: int) -> list[dict[str, float]]:
        """Run `steps` train steps, returning per-step metrics."""
        return [self.train_step(buffer) for _ in range(steps)]

    def save_checkpoint(self, path: str | Path) -> None:
        """Save net weights + optimizer state + step + configs.

        Saved blob is enough to fully resume training from this point.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, Any] = {
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "trainer_config": self.config,
            "pvnet_config": self.net.config,
        }
        torch.save(blob, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "Trainer":
        """Reconstruct a Trainer (net + optimizer + step) from a checkpoint."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        pvnet_config: PVNetConfig = blob["pvnet_config"]
        trainer_config: TrainerConfig = blob["trainer_config"]

        net = PVNet(pvnet_config)
        net.load_state_dict(blob["net_state_dict"])
        trainer = cls(net, config=trainer_config)
        trainer.optimizer.load_state_dict(blob["optimizer_state_dict"])
        trainer.step_count = blob["step_count"]
        return trainer
