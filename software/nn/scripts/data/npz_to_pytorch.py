#!/usr/bin/env python3
"""Convert experiment model artifacts to PyTorch .pt checkpoints.

Supported inputs:
- .npz files (if present)
- *_state_dict.txt and *_params.txt text dumps
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Any

import numpy as np


def convert_npz_to_payload(npz_path: pathlib.Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    tensors: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    for key in data.files:
        arr = data[key]
        if arr.dtype.kind in {"U", "S"}:
            metadata[key] = arr.tolist()
            continue
        tensors[key] = arr

    payload: Dict[str, Any] = {"state_dict": tensors}
    if metadata:
        payload["metadata"] = metadata
    return payload


def convert_text_dump_to_payload(txt_path: pathlib.Path) -> Dict[str, Any]:
    state_dict: Dict[str, Any] = {}
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            name = parts[0]
            rows = int(parts[1])
            cols = int(parts[2])
            values = [float(v) for v in parts[3:]]
            arr = np.asarray(values, dtype=np.float32).reshape(rows, cols)
            state_dict[name] = arr

    return {"state_dict": state_dict}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert experiment checkpoints to PT files.")
    parser.add_argument(
        "--models-dir",
        default="results/guayaquil/models",
        help="Directory to scan recursively for model artifacts.",
    )
    args = parser.parse_args()

    models_dir = pathlib.Path(args.models_dir)
    if not models_dir.exists():
        print(f"[npz->pt] models directory not found: {models_dir}")
        return 1

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"[npz->pt] torch import failed: {exc}")
        return 1

    converted = 0

    for txt_path in sorted(models_dir.rglob("*_state_dict.txt")):
        payload = convert_text_dump_to_payload(txt_path)
        pt_path = txt_path.with_suffix(".pt")
        torch.save(payload, pt_path)
        converted += 1
        print(f"[artifact->pt] wrote: {pt_path}")

    for txt_path in sorted(models_dir.rglob("*_params.txt")):
        payload = convert_text_dump_to_payload(txt_path)
        pt_path = txt_path.with_suffix(".pt")
        torch.save(payload, pt_path)
        converted += 1
        print(f"[artifact->pt] wrote: {pt_path}")

    for npz_path in sorted(models_dir.rglob("*.npz")):
        payload = convert_npz_to_payload(npz_path)
        pt_path = npz_path.with_suffix(".pt")
        torch.save(payload, pt_path)
        converted += 1
        print(f"[artifact->pt] wrote: {pt_path}")

    print(f"[artifact->pt] converted files: {converted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
