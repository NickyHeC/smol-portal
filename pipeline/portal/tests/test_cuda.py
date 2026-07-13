"""Tests for smolvm CUDA configuration helpers."""

from __future__ import annotations

from portal.cuda import causal_lm_load_kwargs, configure_cuda_for_smolvm


def test_configure_cuda_for_smolvm_no_cuda():
    configure_cuda_for_smolvm()


def test_configure_cuda_for_smolvm_skip_sdpa_overrides(monkeypatch):
    monkeypatch.setenv("PORTAL_SKIP_CUDA_SMOLVM", "1")
    import torch

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    configure_cuda_for_smolvm()

    assert torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()
    assert not torch.backends.cuda.math_sdp_enabled()


def test_causal_lm_load_kwargs_cpu():
    kwargs = causal_lm_load_kwargs()
    assert kwargs["trust_remote_code"] is True
    assert "device_map" not in kwargs
