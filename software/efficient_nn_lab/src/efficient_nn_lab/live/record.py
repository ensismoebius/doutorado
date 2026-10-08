"""CLI: record a few short labeled clips per vowel for `train.py`.

Run as ``python -m efficient_nn_lab.live.record --vowel a --seconds 2 --takes 6``
(repeat per vowel, a/e/i/o/u) before the talk. Clips are saved as plain
``.npy`` under ``data/vowel_snn/raw/<vowel>/`` (gitignored -- it's your own
voice, no reason for it to leave your machine).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "vowel_snn" / "raw"
_SAMPLE_RATE = 16000


def record_clip(seconds: float, sample_rate: int) -> np.ndarray:
    import sounddevice as sd  # lazy: this module is only run manually, not imported by the app

    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vowel", required=True, choices=["a", "e", "i", "o", "u"])
    parser.add_argument("--seconds", type=float, default=2.0, help="length of each take")
    parser.add_argument("--takes", type=int, default=6, help="how many separate recordings to make")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between takes, to re-read the prompt")
    parser.add_argument("--sample-rate", type=int, default=_SAMPLE_RATE)
    args = parser.parse_args()

    out_dir = _DATA_ROOT / args.vowel
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("clip_*.npy")))

    for i in range(args.takes):
        clip_index = existing + i
        input(f"Vogal '{args.vowel}', take {i + 1}/{args.takes}: Enter para gravar {args.seconds:g}s...")
        audio = record_clip(args.seconds, args.sample_rate)
        out_path = out_dir / f"clip_{clip_index:03d}.npy"
        np.save(out_path, audio)
        print(f"  salvo em {out_path}")
        time.sleep(args.pause)

    print(f"Pronto: {args.takes} novas gravações de '{args.vowel}' em {out_dir}")


if __name__ == "__main__":
    main()
