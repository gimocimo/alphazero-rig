# alphazero-rig

A from-scratch AlphaZero implementation built as a **learning rig**, then pushed toward a **novel non-game application**.

Inspired by Eric Jang's argument (Dwarkesh interview) that the way to internalise a model is to rebuild it.

## Philosophy

- **End-to-end first, quality second.** Get a working self-play loop on a trivial game before optimising any single component. Many AlphaZero attempts die because someone spent a month perfecting MCTS without ever closing the training loop.
- **Small games before big games.** Connect4 → Othello → 9×9 Go. Full 19×19 Go is not solo-feasible — don't aim for it.
- **The rig is the artifact.** Components are reusable across games and (eventually) non-game domains. The interface boundaries matter more than any single implementation.

## Roadmap

### Phase 1 — Foundation (complete)
- [x] Project scaffolding
- [x] `Game` interface + Connect4
- [x] Random-vs-random self-play harness
- [x] Pure MCTS (no neural net) — beats random 9/10 with 100 sims
- [x] PVNet — small ResNet policy/value network (~60k params)
- [x] PUCT search using the network's priors
- [x] Self-play data generation → replay buffer
- [x] Trainer with MPS auto-detection + checkpointing
- [x] Evaluation arena (alternating sides, draws-as-half winrate)
- [x] Closed-loop training driver (`python -m alphazero.train`)

### Phase 2 — Scaling up
- [ ] Othello (8×8) — same rig, new game
- [ ] 9×9 Go — same rig, new game
- [ ] Distributed self-play (multiprocess)
- [ ] Network: ResNet-style backbone, dual heads
- [ ] Training tricks: data augmentation via symmetries, Dirichlet noise, temperature schedule

### Phase 3 — Novel application
The interesting part. Candidates (to decide once Phase 1 works):
- **MuZero-style learned dynamics** on a non-game domain (e.g. small scheduling problem, a digital twin).
- **AlphaZero on a safe-RL toy problem** — incorporate constraint satisfaction into the search.
- **Search-augmented agent** — use MCTS-style planning over tool calls or task decompositions.

The phase 3 choice depends on what surprises us in phases 1–2.

## Layout

```
alphazero/
  agents/        # Pluggable agents (Random, MCTS, PUCT+net)
  games/         # Game environments (Game ABC + Connect4)
  mcts/          # MCTS variants — pure_mcts, puct (network-guided)
  nets/          # PVNet — policy + value network
  training/      # ReplayBuffer, self-play, Trainer, train loop
  eval/          # Arena (head-to-head winrate)
  train.py       # CLI entry: `python -m alphazero.train`
tests/           # pytest suite
runs/            # Training outputs (gitignored)
```

## Development

```bash
uv sync --extra dev
uv run pytest
```

## Training

Run the full AlphaZero loop:

```bash
uv run python -m alphazero.train
```

Defaults are tuned for Connect4 on a laptop (100 iterations × 40 self-play
games × 80 MCTS sims/move). On Apple Silicon the device is auto-detected
as MPS. See `--help` for tunable hyperparameters.

Quick smoke run (a few minutes):

```bash
uv run python -m alphazero.train \
    --iterations 5 \
    --games-per-iteration 10 \
    --simulations 40 \
    --train-steps-per-iteration 50 \
    --batch-size 32 \
    --min-buffer-size 100 \
    --eval-every 2 \
    --eval-games 8 \
    --output-dir runs/smoke
```

Each run writes:
- `metrics.csv` — one row per iteration: losses, eval winrate, timing.
- `latest.pt` — checkpoint after the most recent iteration. Use to resume.
- `best.pt` — last promoted net (a candidate must beat the current best
  by ≥55% winrate to promote).

Resume an interrupted run:

```bash
uv run python -m alphazero.train --resume runs/smoke/latest.pt
```
