#!/usr/bin/env python3
"""
Convert all .mat files in a directory to compressed .npz files.

Usage:
    python3 scripts/mat_to_npz.py /path/to/mat_dir /path/to/out_dir

Note: Requires scipy (scipy.io.loadmat) and numpy.
"""
import sys
import os
from pathlib import Path

def convert_file(mat_path: Path, out_dir: Path):
    try:
        from scipy.io import loadmat
        import numpy as np
    except Exception as e:
        print("Missing dependency: install scipy and numpy to use this script")
        raise

    data = loadmat(str(mat_path))
    # Remove MATLAB metadata keys starting with '__'
    data = {k: v for k, v in data.items() if not k.startswith('__')}
    out_path = out_dir / (mat_path.stem + '.npz')
    np.savez_compressed(str(out_path), **data)
    print(f"Converted {mat_path} -> {out_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: mat_to_npz.py /path/to/mat_dir /path/to/out_dir")
        sys.exit(1)

    mat_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    mats = list(mat_dir.glob('**/*.mat'))
    if not mats:
        print("No .mat files found in", mat_dir)
        return

    for m in mats:
        try:
            convert_file(m, out_dir)
        except Exception as e:
            print(f"Failed to convert {m}: {e}")


if __name__ == '__main__':
    main()
