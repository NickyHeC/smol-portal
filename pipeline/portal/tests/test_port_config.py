"""Tests that `portal port` sizing knobs propagate into per-stage configs."""

from __future__ import annotations

from portal.config import LatentMode, PortConfig


def _base(**kw) -> PortConfig:
    return PortConfig(
        source_model="src",
        target_model="tgt",
        task_name="t",
        dataset_name="ds",
        **kw,
    )


def test_defaults_match_stage_defaults():
    # A plain port run must reproduce the previous stage-config defaults so
    # content-addressed artifacts don't drift.
    cfg = _base()
    train = cfg.build_train_config()
    assert train.num_epochs == 3
    assert train.batch_size == 4
    assert train.max_seq_length == 512
    assert train.max_samples is None
    assert train.lora.rank == 16

    hyper = cfg.build_hypernet_config()
    assert hyper.latent_dim == 256
    assert hyper.hidden_dim == 512
    assert hyper.num_epochs == 50
    assert hyper.num_layers == 3

    conv = cfg.build_converter_config()
    assert conv.calibration_dataset == "ds"  # falls back to dataset_name
    assert conv.calibration_samples == 256
    assert conv.num_epochs == 30
    assert conv.latent_mode == LatentMode.REAL


def test_knobs_propagate_into_all_stages():
    cfg = _base(
        max_samples=64,
        max_seq_length=128,
        batch_size=1,
        lora_rank=8,
        train_epochs=1,
        extract_epochs=10,
        convert_epochs=20,
        cal_samples=32,
        latent_dim=64,
        hidden_dim=128,
        latent_mode=LatentMode.RANDOM,
        seed=7,
    )

    train = cfg.build_train_config()
    assert train.max_samples == 64
    assert train.max_seq_length == 128
    assert train.batch_size == 1
    assert train.lora.rank == 8
    assert train.num_epochs == 1
    assert train.seed == 7

    hyper = cfg.build_hypernet_config()
    assert hyper.latent_dim == 64
    assert hyper.hidden_dim == 128
    assert hyper.num_epochs == 10
    assert hyper.seed == 7

    conv = cfg.build_converter_config()
    assert conv.calibration_samples == 32
    assert conv.num_epochs == 20
    assert conv.hidden_dim == 128
    assert conv.latent_mode == LatentMode.RANDOM
    assert conv.seed == 7

    ev = cfg.build_eval_config("tgt")
    assert ev.max_samples == 64
    assert ev.batch_size == 1
    assert ev.max_seq_length == 128


def test_cal_dataset_override():
    cfg = _base(calibration_dataset="other-ds")
    assert cfg.build_converter_config().calibration_dataset == "other-ds"


def test_explicit_stage_config_overrides_knobs():
    from portal.config import ConverterConfig

    explicit = ConverterConfig(target_model="tgt", calibration_dataset="ds", num_epochs=99)
    cfg = _base(converter=explicit, convert_epochs=20)
    # Explicit config wins over the knob.
    assert cfg.build_converter_config().num_epochs == 99
