"""Tests for converter latent-ablation transforms."""

from __future__ import annotations

import torch

from portal.config import LatentMode
from portal.converter import apply_latent_mode


def test_real_is_identity():
    z = torch.randn(16)
    out = apply_latent_mode(z, LatentMode.REAL, seed=0)
    assert torch.equal(out, z)


def test_zero_mode():
    z = torch.randn(16)
    out = apply_latent_mode(z, LatentMode.ZERO, seed=0)
    assert torch.count_nonzero(out) == 0
    assert out.shape == z.shape


def test_random_mode_is_deterministic_and_differs():
    z = torch.randn(32)
    a = apply_latent_mode(z, LatentMode.RANDOM, seed=7)
    b = apply_latent_mode(z, LatentMode.RANDOM, seed=7)
    assert torch.equal(a, b)  # deterministic given seed
    assert not torch.equal(a, z)  # actually replaces the latent
    assert a.shape == z.shape


def test_shuffled_is_permutation():
    z = torch.arange(64, dtype=torch.float32)
    out = apply_latent_mode(z, LatentMode.SHUFFLED, seed=3)
    assert out.shape == z.shape
    # A permutation preserves the multiset of values but (almost surely) reorders.
    assert torch.equal(torch.sort(out).values, torch.sort(z).values)
    assert not torch.equal(out, z)
