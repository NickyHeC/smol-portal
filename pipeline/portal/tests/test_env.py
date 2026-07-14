"""Tests for the runtime provenance manifest."""

from __future__ import annotations

from portal.env import runtime_manifest


def test_manifest_has_core_fields():
    m = runtime_manifest()
    assert "portal_version" in m
    assert "python" in m
    assert "platform" in m
    assert "packages" in m
    assert "git_commit" in m  # may be None inside the VM; key must exist


def test_manifest_tracks_key_packages():
    packages = runtime_manifest()["packages"]
    for name in ("torch", "transformers", "peft"):
        assert name in packages


def test_manifest_is_json_serialisable():
    import json

    json.dumps(runtime_manifest())
