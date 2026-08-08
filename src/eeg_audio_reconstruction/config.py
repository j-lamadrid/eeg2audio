"""Configuration dataclasses used by scripts and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    hdf5_path: str = "data/madeeg_preprocessed.hdf5"
    metadata_path: str = "data/madeeg_preprocessed.yaml"
    segment_seconds: float = 1.0
    eeg_sample_rate: int = 256
    average_repetitions: int = 4
    target: str = "stimulus"
    normalize_eeg: bool = True
    split: str | None = "train"
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    max_subjects: int | None = None
    max_stimuli_per_subject: int | None = None
    max_samples: int | None = None


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    frames: int = 64
    f_min: float = 20.0
    f_max: float | None = None
    top_db: float = 80.0


@dataclass
class AutoencoderConfig:
    latent_channels: int = 8
    base_channels: int = 32


@dataclass
class DenoiserConfig:
    model_channels: int = 96
    cond_dim: int = 256
    num_res_blocks: int = 6


@dataclass
class DiffusionConfig:
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    target_alpha_bar: float | None = None


def dataclass_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
