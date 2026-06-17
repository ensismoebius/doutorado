"""Configuração central do protótipo multimodal.

Mantém os parâmetros definidos em PLANNING.md e um contrato de shapes compatível
com a arquitetura C++ do projeto (batch-first e variante temporal T*B,F).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    format: Literal["mat", "npz", "wav_csv"] = "npz"
    audio_key: str = "audio"
    eeg_key: str = "eeg"
    speaker_key: str = "speaker_id"
    pair_index_key: str = "eeg_index"
    sample_key: str = "sample_id"


@dataclass(frozen=True)
class PreprocessConfig:
    window_sec: float = 0.1
    overlap: float = 0.5
    target_audio_sr: int = 16_000
    target_eeg_sr: int = 200
    zscore_per_window: bool = True


@dataclass(frozen=True)
class AEConfig:
    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (512, 256)
    model_type: Literal["dense", "spiking"] = "spiking"
    snn_time_steps: int = 5
    snn_dt: float = 1e-3
    snn_resistance: float = 5.0
    snn_capacitance: float = 1.0
    snn_threshold: float = 1.0
    snn_surrogate_sharpness: float = 1.0


@dataclass(frozen=True)
class WaveletConfig:
    family: str = "db4"
    max_level: int = 4


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    save_features: bool = True


@dataclass(frozen=True)
class PrototypeConfig:
    data: DataConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ae: AEConfig = field(default_factory=AEConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(
            output_dir=Path("outputs/prototipo_multimodal")
        )
    )
