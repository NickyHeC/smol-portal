"""Tests for the latent → adapter converter training mechanics."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call

from portal.converter import LatentToLoraConverter


class _TinyModel(nn.Module):
    """Minimal stand-in for a PeftModel: one named weight used in the forward."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.weight).sum()


def test_functional_call_keeps_converter_in_graph():
    """Gradients must flow back to the converter through functional_call.

    Regression guard for the bug where predicted weights were written to
    ``param.data`` (detached), leaving the converter untrained.
    """
    dim = 8
    model = _TinyModel(dim)
    converter = LatentToLoraConverter(latent_dim=4, output_dim=dim, hidden_dim=16)
    z = torch.randn(1, 4)

    predicted = converter(z).squeeze(0)
    out = functional_call(model, {"weight": predicted}, args=(torch.ones(dim),))
    out.backward()

    grads = [p.grad for p in converter.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_data_copy_detaches_graph():
    """Documents the old (broken) path: .data.copy_ severs the graph."""
    dim = 8
    model = _TinyModel(dim)
    converter = LatentToLoraConverter(latent_dim=4, output_dim=dim, hidden_dim=16)
    z = torch.randn(1, 4)

    predicted = converter(z).squeeze(0)
    model.weight.data.copy_(predicted.detach())
    model(torch.ones(dim)).backward()

    assert all(p.grad is None for p in converter.parameters())
