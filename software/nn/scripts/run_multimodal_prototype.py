#!/usr/bin/env python3
"""CLI para executar o protótipo multimodal EEG+Áudio."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.demos.pydemos.prototipo_multimodal.config import (
    AEConfig,
    DataConfig,
    OutputConfig,
    PreprocessConfig,
    PrototypeConfig,
    TrainConfig,
    WaveletConfig,
)
from src.demos.pydemos.prototipo_multimodal.run_prototype import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protótipo multimodal com PyTorch+snnTorch")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--format", choices=["npz", "mat", "wav_csv"], default="npz")
    p.add_argument("--audio-orig-sr", type=int, required=True)
    p.add_argument("--eeg-orig-sr", type=int, required=True)
    p.add_argument("--model", choices=["dense", "spiking"], default="spiking")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/prototipo_multimodal")
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PrototypeConfig(
        data=DataConfig(data_root=args.data_root, format=args.format),
        preprocess=PreprocessConfig(overlap=args.overlap),
        ae=AEConfig(model_type=args.model, latent_dim=args.latent_dim),
        wavelet=WaveletConfig(),
        train=TrainConfig(epochs=args.epochs, batch_size=args.batch_size),
        output=OutputConfig(output_dir=args.output_dir),
    )

    summary = run(
        cfg=cfg, audio_orig_sr=args.audio_orig_sr, eeg_orig_sr=args.eeg_orig_sr
    )
    print("Resumo do protótipo:")
    print(summary)


if __name__ == "__main__":
    main()
