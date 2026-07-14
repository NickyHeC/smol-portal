"""Runtime provenance manifest for PorTAL artifacts.

Content-addressing hashes only the *config* (see ``config.content_hash``), which
is what makes reruns idempotent — but a config hash does not capture the library
versions, git commit, or platform that actually produced an artifact. When a run
misbehaves (or a result can't be reproduced), those are exactly what you need.

This module records that environment as a side-channel manifest embedded in
artifact metadata. It is deliberately **excluded from the content hash** so the
same config still resolves to the same artifact directory across machines; the
manifest is provenance, not identity.
"""

from __future__ import annotations

import platform
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Packages whose versions materially change training/eval numerics.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "datasets",
    "safetensors",
    "accelerate",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _git_commit() -> str | None:
    """Best-effort git commit of the portal checkout (None inside the VM/no git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_manifest() -> dict:
    """Return a JSON-serialisable manifest of the current runtime environment."""
    from portal import __version__

    return {
        "portal_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
        "git_commit": _git_commit(),
    }
