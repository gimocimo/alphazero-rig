"""Tests for the replay buffer."""

from __future__ import annotations

import numpy as np
import pytest

from alphazero.training.replay_buffer import ReplayBuffer, TrainingExample


def _make_example(value: float = 0.0) -> TrainingExample:
    return TrainingExample(
        encoded_state=np.zeros((2, 6, 7), dtype=np.float32),
        policy_target=np.full(7, 1 / 7, dtype=np.float32),
        value_target=value,
    )


def test_empty_buffer_has_length_zero() -> None:
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0


def test_add_and_extend() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.add(_make_example(0.5))
    assert len(buf) == 1
    buf.extend([_make_example(0.0), _make_example(-0.5)])
    assert len(buf) == 3


def test_fifo_eviction_at_capacity() -> None:
    buf = ReplayBuffer(capacity=3)
    for v in (0.1, 0.2, 0.3, 0.4, 0.5):
        buf.add(_make_example(v))
    assert len(buf) == 3
    # Oldest entries (0.1, 0.2) should have been evicted; remaining are 0.3..0.5.
    values = sorted(e.value_target for e in buf._buf)
    assert values == [0.3, 0.4, 0.5]


def test_sample_returns_correct_shapes() -> None:
    buf = ReplayBuffer(capacity=100, seed=0)
    for v in np.linspace(-1, 1, 50):
        buf.add(_make_example(float(v)))
    states, policies, values = buf.sample(batch_size=8)
    assert states.shape == (8, 2, 6, 7)
    assert policies.shape == (8, 7)
    assert values.shape == (8,)
    assert states.dtype == np.float32
    assert policies.dtype == np.float32
    assert values.dtype == np.float32


def test_sample_without_replacement_no_duplicates() -> None:
    # Distinct values so we can detect duplicate sampling.
    buf = ReplayBuffer(capacity=100, seed=0)
    for i in range(20):
        buf.add(_make_example(float(i)))
    _, _, values = buf.sample(batch_size=20)
    assert len(set(values.tolist())) == 20, "sample drew duplicates"


def test_sample_raises_when_insufficient() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.add(_make_example())
    with pytest.raises(ValueError, match="need"):
        buf.sample(batch_size=5)


def test_sample_reproducible_with_seed() -> None:
    buf_a = ReplayBuffer(capacity=100, seed=42)
    buf_b = ReplayBuffer(capacity=100, seed=42)
    for v in range(30):
        buf_a.add(_make_example(float(v)))
        buf_b.add(_make_example(float(v)))
    _, _, values_a = buf_a.sample(batch_size=8)
    _, _, values_b = buf_b.sample(batch_size=8)
    assert np.array_equal(values_a, values_b)


def test_capacity_validation() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=0)
