"""CLI: replay a single game between two agents and print the trace.

Examples:
    # Random vs Random.
    uv run python -m alphazero.replay --agent random

    # Pure MCTS at 100 sims, both sides.
    uv run python -m alphazero.replay --agent mcts --simulations 100

    # PUCT loading a trained checkpoint, both sides.
    uv run python -m alphazero.replay \\
        --agent puct \\
        --checkpoint runs/connect4/best.pt \\
        --simulations 200
"""

from __future__ import annotations

import argparse

import torch

from .agents.base import Agent
from .agents.random_agent import RandomAgent
from .eval.replay import render_game_replay
from .games.connect4 import Connect4
from .mcts.puct import PUCTAgent
from .mcts.pure_mcts import MCTSAgent
from .training.trainer import Trainer


def _build_agent(
    kind: str,
    simulations: int,
    checkpoint: str | None,
    device: str | None,
    seed: int,
) -> Agent:
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "mcts":
        return MCTSAgent(simulations=simulations, seed=seed)
    if kind == "puct":
        if checkpoint is None:
            # Use a random-init net — useful for verifying the pipeline.
            from .nets.pvnet import PVNet

            net = PVNet()
        else:
            net = Trainer.from_checkpoint(checkpoint).net
        device_str = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        net = net.to(device_str)
        return PUCTAgent(
            net,
            simulations=simulations,
            dirichlet_epsilon=0.0,  # noise off for evaluation/replay
            device=device_str,
            seed=seed,
        )
    raise ValueError(f"unknown agent kind {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a single game between two agents with per-ply diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=("random", "mcts", "puct"),
        default="random",
        help="Agent class for both sides",
    )
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="For --agent puct: path to .pt checkpoint; random-init net if omitted",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top children to show per ply"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    game = Connect4()
    agent_a = _build_agent(args.agent, args.simulations, args.checkpoint, args.device, args.seed)
    agent_b = _build_agent(
        args.agent, args.simulations, args.checkpoint, args.device, args.seed + 1
    )

    print(render_game_replay(game, agent_a, agent_b, top_k_per_ply=args.top_k))


if __name__ == "__main__":
    main()
