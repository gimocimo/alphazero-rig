"""The AlphaZero training loop.

One iteration of the loop:
    1. Self-play: current net + PUCT plays `games_per_iteration` games
       against itself, appending all positions/visits/outcomes to the
       replay buffer.
    2. Training: `train_steps_per_iteration` SGD-with-Adam updates on
       minibatches sampled from the buffer.
    3. Evaluation (every `eval_every` iterations): current net plays
       `eval_games` games against the current "best" net checkpoint.
       If winrate > `promote_threshold`, promote current as the new best.
    4. Logging: append one row to metrics.csv with losses + evaluation
       outcome + timing.
    5. Checkpoint: save current net + optimizer state to `latest.pt`.

Why a "best" checkpoint?
    Training is non-monotonic — a step that fixes one position may regress
    another. We want to ship a net that is *demonstrably stronger* than
    earlier ones, not whichever was most recent. The arena-based gating
    is exactly the AlphaZero original design.

Configurable via TrainingLoopConfig. Sensible defaults are tuned for
Connect4 on a laptop; smaller games (Connect3) or larger (9x9 Go) need
different scaling.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..eval.arena import arena_play
from ..games.connect4 import Connect4
from ..mcts.puct import PUCTAgent
from ..nets.pvnet import PVNet
from .replay_buffer import ReplayBuffer
from .self_play import play_self_play_game
from .trainer import Trainer, TrainerConfig


@dataclass
class TrainingLoopConfig:
    # Iteration budget.
    iterations: int = 100
    games_per_iteration: int = 40
    train_steps_per_iteration: int = 200

    # Search settings during self-play (exploration enabled).
    self_play_simulations: int = 80
    dirichlet_alpha: float = 0.5
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 8  # plies of T=1 before switching to T=0

    # Buffer.
    buffer_capacity: int = 50_000
    min_buffer_size: int = 500  # don't bother training before this

    # Optimizer.
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    grad_clip: Optional[float] = 1.0

    # Evaluation against current best (noise-free, low temperature).
    eval_every: int = 5
    eval_games: int = 20
    eval_simulations: int = 80
    promote_threshold: float = 0.55

    # Periodic dated snapshots (in addition to latest.pt and best.pt).
    # Each snapshot is saved as iter_NNNN.pt so opening-discovery and other
    # over-time analyses can iterate a directory. 0 disables.
    snapshot_every: int = 10

    # I/O.
    output_dir: str = "runs/connect4"
    device: Optional[str] = None  # auto-detect
    seed: int = 0


_METRIC_FIELDS = [
    "iteration",
    "timestamp",
    "elapsed_seconds",
    "buffer_size",
    "train_steps_total",
    "mean_loss",
    "mean_loss_policy",
    "mean_loss_value",
    "self_play_seconds",
    "train_seconds",
    "eval_winrate",
    "promoted",
]


def _build_self_play_agent(
    net: PVNet, cfg: TrainingLoopConfig, device: str, seed: int
) -> PUCTAgent:
    return PUCTAgent(
        net,
        simulations=cfg.self_play_simulations,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        device=device,
        seed=seed,
    )


def _build_eval_agent(net: PVNet, cfg: TrainingLoopConfig, device: str, seed: int) -> PUCTAgent:
    # No noise; temperature=0 is set at the call site.
    return PUCTAgent(
        net,
        simulations=cfg.eval_simulations,
        dirichlet_epsilon=0.0,
        device=device,
        seed=seed,
    )


def train_loop(
    config: TrainingLoopConfig,
    resume_from: Optional[str | Path] = None,
) -> None:
    """Run the AlphaZero training loop.

    If `resume_from` is given, the net + optimizer + step count are
    restored from that checkpoint. The replay buffer is *not* persisted
    and will refill from scratch — practical because a couple of self-play
    iterations rebuild it quickly.
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    game = Connect4()

    # Net + trainer.
    if resume_from is not None:
        trainer = Trainer.from_checkpoint(resume_from)
        net = trainer.net
        print(f"Resumed from {resume_from} at step {trainer.step_count}")
    else:
        net = PVNet()
        trainer = Trainer(
            net,
            TrainerConfig(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                batch_size=config.batch_size,
                grad_clip=config.grad_clip,
                device=config.device,
            ),
        )

    # The "best" net starts as a copy of the current one. We compare against
    # it for promotion decisions. Save it now so `best.pt` always exists,
    # even if no promotion ever happens (useful for downstream tooling).
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    trainer.save_checkpoint(out / "best.pt")
    last_promoted_iteration: Optional[int] = None

    buffer = ReplayBuffer(capacity=config.buffer_capacity, seed=config.seed)

    # Initialise (or extend) the CSV log.
    metrics_path = out / "metrics.csv"
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_METRIC_FIELDS)
        if write_header:
            writer.writeheader()

    print(f"Device: {trainer.device}  |  params: {net.num_parameters():,}  |  output: {out}")
    start_time = time.time()

    for iteration in range(1, config.iterations + 1):
        iter_seed = config.seed + iteration * 1000

        # --- Self-play ---
        sp_start = time.time()
        sp_agent = _build_self_play_agent(net, config, trainer.device, iter_seed)
        for _ in range(config.games_per_iteration):
            examples = play_self_play_game(
                game, sp_agent, temperature_threshold=config.temperature_threshold
            )
            buffer.extend(examples)
        sp_time = time.time() - sp_start

        # --- Training ---
        tr_start = time.time()
        losses_p: list[float] = []
        losses_v: list[float] = []
        losses_t: list[float] = []
        # Need at least one full minibatch *and* the user's warmup threshold.
        train_gate = max(config.min_buffer_size, config.batch_size)
        if len(buffer) >= train_gate:
            for _ in range(config.train_steps_per_iteration):
                m = trainer.train_step(buffer)
                losses_p.append(m["loss_policy"])
                losses_v.append(m["loss_value"])
                losses_t.append(m["loss"])
        tr_time = time.time() - tr_start

        # --- Evaluation ---
        eval_winrate: Optional[float] = None
        promoted = 0
        if iteration % config.eval_every == 0 and len(buffer) >= train_gate:
            best_net = PVNet(net.config)
            best_net.load_state_dict(best_state)
            best_net = best_net.to(trainer.device)

            current_agent = _build_eval_agent(net, config, trainer.device, iter_seed + 7)
            best_agent = _build_eval_agent(best_net, config, trainer.device, iter_seed + 11)
            result = arena_play(
                game, current_agent, best_agent, num_games=config.eval_games
            )
            eval_winrate = result.agent_a_winrate

            if eval_winrate > config.promote_threshold:
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                trainer.save_checkpoint(out / "best.pt")
                last_promoted_iteration = iteration
                promoted = 1

        # --- Logging ---
        elapsed = time.time() - start_time
        mean_t = float(np.mean(losses_t)) if losses_t else float("nan")
        mean_p = float(np.mean(losses_p)) if losses_p else float("nan")
        mean_v = float(np.mean(losses_v)) if losses_v else float("nan")

        row = {
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": f"{elapsed:.1f}",
            "buffer_size": len(buffer),
            "train_steps_total": trainer.step_count,
            "mean_loss": f"{mean_t:.4f}",
            "mean_loss_policy": f"{mean_p:.4f}",
            "mean_loss_value": f"{mean_v:.4f}",
            "self_play_seconds": f"{sp_time:.1f}",
            "train_seconds": f"{tr_time:.1f}",
            "eval_winrate": f"{eval_winrate:.3f}" if eval_winrate is not None else "",
            "promoted": promoted,
        }
        with metrics_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=_METRIC_FIELDS).writerow(row)

        eval_str = (
            f"  eval={eval_winrate:.2f}{' PROMOTED' if promoted else ''}"
            if eval_winrate is not None
            else ""
        )
        print(
            f"iter {iteration:3d}/{config.iterations}  "
            f"buf={len(buffer):5d}  "
            f"loss={mean_t:.3f} (p={mean_p:.3f} v={mean_v:.3f})  "
            f"sp={sp_time:5.1f}s tr={tr_time:5.1f}s"
            f"{eval_str}"
        )

        trainer.save_checkpoint(out / "latest.pt")
        if config.snapshot_every > 0 and iteration % config.snapshot_every == 0:
            trainer.save_checkpoint(out / f"iter_{iteration:04d}.pt")

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"\nDone. {config.iterations} iterations in {elapsed_min:.1f} min.")
    print(f"Logs:        {metrics_path}")
    print(f"Latest net:  {out / 'latest.pt'}")
    if last_promoted_iteration is None:
        print(f"Best net:    {out / 'best.pt'} (still the initial net — no promotion happened)")
    else:
        print(f"Best net:    {out / 'best.pt'} (last promoted at iter {last_promoted_iteration})")
