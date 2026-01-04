#!/usr/bin/env python3
"""Fail if Eigen usage appears outside an allowlist.

Usage:
  python scripts/check_eigen_leaks.py \
    --allowlist eigen_allowlist.txt \
    --root . \
    --paths src include

The allowlist should contain one path per line, relative to repository root.
"""

import argparse
import pathlib
import re
import sys
from typing import Iterable, List, Set

EIGEN_PATTERN = re.compile(r"(#\s*include\s*<Eigen|Eigen::)")
DEFAULT_EXTS = {".cpp", ".cc", ".c", ".hpp", ".h", ".ipp", ".tpp"}


def load_allowlist(path: pathlib.Path) -> Set[pathlib.Path]:
    if not path.exists():
        raise FileNotFoundError(f"Allowlist file not found: {path}")
    entries: Set[pathlib.Path] = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.add(pathlib.Path(stripped).resolve())
    return entries


def iter_source_files(root: pathlib.Path, paths: Iterable[str]) -> Iterable[pathlib.Path]:
    for rel in paths:
        base = (root / rel).resolve()
        if not base.exists():
            continue
        for file in base.rglob("*"):
            if file.suffix in DEFAULT_EXTS and file.is_file():
                yield file


def scan_file(path: pathlib.Path) -> List[int]:
    hits: List[int] = []
    try:
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if EIGEN_PATTERN.search(line):
                hits.append(i)
    except UnicodeDecodeError:
        # Skip binary/unknown encodings
        return []
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect Eigen usage outside allowlist")
    parser.add_argument("--allowlist", default="eigen_allowlist.txt", type=str)
    parser.add_argument("--root", default=".", type=str)
    parser.add_argument("--paths", nargs="*", default=["src", "include"], help="Relative paths to scan")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    allowlist = load_allowlist(root / args.allowlist)

    violations: List[str] = []
    for file in iter_source_files(root, args.paths):
        hits = scan_file(file)
        if not hits:
            continue
        if file.resolve() in allowlist:
            continue
        rel = file.relative_to(root)
        violations.append(f"{rel}: Eigen usage at lines {hits}")

    if violations:
        print("Eigen leak check FAILED. Offending files:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print("Eigen leak check passed: no usage outside allowlist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
