"""Tests for smolvm CUDA configuration helpers."""

from __future__ import annotations

from portal.cuda import causal_lm_load_kwargs, configure_cuda_for_smolvm


def test_configure_cuda_for_smolvm_no_cuda():
    configure_cuda_for_smolvm()


def test_causal_lm_load_kwargs_cpu():
    kwargs = causal_lm_load_kwargs()
    assert kwargs["trust_remote_code"] is True
    assert "device_map" not in kwargs
