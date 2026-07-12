"""CUDA runtime tweaks for smolvm's remoted GPU stack."""

from __future__ import annotations

import os


def configure_cuda_for_smolvm() -> None:
    """Apply backend settings validated on smolvm + CUDA shim.

    smolvm's CUDA remoting does not support fused SDPA backward kernels yet;
    flash/mem-efficient attention must be disabled so Llama training uses the
    math SDPA path. Also disable cuDNN and PyTorch expandable segments (VMM)
    until the host shim exposes the full surface.
    """
    import torch

    if not torch.cuda.is_available():
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def causal_lm_load_kwargs() -> dict:
    """``from_pretrained`` kwargs that load safely on smolvm CUDA."""
    import torch

    kwargs: dict = {
        "torch_dtype": torch.float32,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        # Incremental placement — bulk model.to("cuda") can fail on remoted CUDA.
        kwargs["device_map"] = "cuda"
    return kwargs
