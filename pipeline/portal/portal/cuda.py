"""CUDA runtime tweaks for smolvm's remoted GPU stack."""

from __future__ import annotations

import os


def configure_cuda_for_smolvm() -> None:
    """Apply backend settings validated on smolvm + CUDA shim.

    By default, flash/mem-efficient SDPA are disabled so Llama training uses the
    math SDPA path (needed on smolvm <1.5.2). Also disable cuDNN and PyTorch
    expandable segments (VMM) until the host shim exposes the full surface.

    Set ``PORTAL_SKIP_CUDA_SMOLVM=1`` to skip the SDPA overrides only (keep
    alloc-conf + cuDNN tweaks). Validated on smolvm v1.5.2+ with fused SDPA
    backward.
    """
    import torch

    if not torch.cuda.is_available():
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

    torch.backends.cudnn.enabled = False

    if os.environ.get("PORTAL_SKIP_CUDA_SMOLVM") == "1":
        return

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
