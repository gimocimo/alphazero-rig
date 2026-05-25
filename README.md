# alphazero-rig

A from-scratch AlphaZero implementation built as a **learning rig**, then pushed toward a **novel non-game application**.

Inspired by Eric Jang's argument (Dwarkesh interview) that the way to internalise a model is to rebuild it.

## Philosophy

- **End-to-end first, quality second.** Get a working self-play loop on a trivial game before optimising any single component. Many AlphaZero attempts die because someone spent a month perfecting MCTS without ever closing the training loop.
- **Small games before big games.** Connect4 → Othello → 9×9 Go. Full 19×19 Go is not solo-feasible — don't aim for it.
- **The rig is the artifact.** Components are reusable across games and (eventually) non-game domains. The interface boundaries matter more than any single implementation.

## Roadmap

### Phase 1 — Foundation (current)
- [x] Project scaffolding
- [ ] `Game` interface + Connect4
- [ ] Random-vs-random self-play harness
- [ ] Pure MCTS (no neural net) — should crush random
- [ ] Tiny policy/value network (PyTorch)
- [ ] PUCT search using the network's priors
- [ ] Self-play data generation → replay buffer → training step
- [ ] Closed-loop training: net trains on its own MCTS rollouts
- [ ] Evaluation: Elo vs prior checkpoints, head-to-head vs baselines

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
  games/      # Game environments (Game ABC + concrete games)
  mcts/       # Monte Carlo Tree Search variants
  nets/       # Policy/value networks
  training/   # Self-play, replay buffer, trainer
  eval/       # Elo, arenas, head-to-head
  agents/     # Pluggable agents (Random, MCTS, PUCT+net)
tests/        # pytest suite
```

## Development

```bash
uv sync --extra dev
uv run pytest
```
