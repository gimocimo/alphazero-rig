"""CLI entry point for training.

Run:
    uv run python -m alphazero.train

The defaults are tuned for Connect4 on a laptop. Run --help to see all
options. Power users can edit `TrainingLoopConfig` in
alphazero/training/loop.py directly for hyperparameters not exposed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .training.loop import TrainingLoopConfig, train_loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an AlphaZero agent on Connect4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = TrainingLoopConfig()

    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument(
        "--games-per-iteration", type=int, default=defaults.games_per_iteration
    )
    parser.add_argument(
        "--train-steps-per-iteration",
        type=int,
        default=defaults.train_steps_per_iteration,
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=defaults.self_play_simulations,
        help="Self-play MCTS simulations per move",
    )
    parser.add_argument(
        "--eval-simulations", type=int, default=defaults.eval_simulations
    )
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-games", type=int, default=defaults.eval_games)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--buffer-capacity", type=int, default=defaults.buffer_capacity
    )
    parser.add_argument(
        "--min-buffer-size", type=int, default=defaults.min_buffer_size
    )
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override auto-detection: cpu, mps, or cuda",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=defaults.snapshot_every,
        help="Save a dated checkpoint (iter_NNNN.pt) every N iterations; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from",
    )

    args = parser.parse_args()

    config = TrainingLoopConfig(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        train_steps_per_iteration=args.train_steps_per_iteration,
        self_play_simulations=args.simulations,
        eval_simulations=args.eval_simulations,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        buffer_capacity=args.buffer_capacity,
        min_buffer_size=args.min_buffer_size,
        snapshot_every=args.snapshot_every,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    resume_path = Path(args.resume) if args.resume else None
    train_loop(config, resume_from=resume_path)


if __name__ == "__main__":
    main()
