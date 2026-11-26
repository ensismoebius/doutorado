#!/usr/bin/env python3
"""thesis_compare_rebuild.py -- is a rebuilt file the same result as the committed one?

Why. thesis_rebuild_from_results.sh regenerates every file the thesis derives from its runs and
must tell a changed result from two things that are not results:

  * wall-clock fields: the calibration records how long it took ("seconds"), which differs at
    every run on every machine;
  * the last bits of a float: a mean of the same numbers summed in another order (another
    numpy, another CPU) can differ in the 16th significant digit, e.g.
        committed  "r": 0.7091564366639
        rebuilt    "r": 0.7091564366639003        relative difference 4e-16

A byte comparison calls both "different"; ignoring every number would hide a real change. So a
JSON file is compared value by value: the same keys, the same lengths, equal strings, and
numbers within a relative tolerance the caller states. Any other file must be byte-identical.

Usage:
  thesis_compare_rebuild.py --rel-tol 1e-12 [--ignore-key seconds] COMMITTED REBUILT

Prints one line ("identical", "equal within rel-tol ..." or the first differences) and exits 0
when the two are the same result, 1 when they are not. Every option is required to be stated:
there is no default tolerance.
"""
import argparse
import json
import math
import pathlib
import sys


def differences(a, b, path, rel_tol, ignore, out):
    """Append to out a description of every place where a and b differ beyond rel_tol."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a, keys_b = set(a) - ignore, set(b) - ignore
        for k in sorted(keys_a ^ keys_b):
            out.append(f"{path}/{k}: present on one side only")
        for k in sorted(keys_a & keys_b):
            differences(a[k], b[k], f"{path}/{k}", rel_tol, ignore, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            differences(x, y, f"{path}[{i}]", rel_tol, ignore, out)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                out.append(f"{path}: {a} vs {b}")
        elif math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                out.append(f"{path}: {a!r} vs {b!r}")
        elif not math.isclose(a, b, rel_tol=rel_tol, abs_tol=0.0):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rel-tol", required=True, type=float,
                    help="relative tolerance for numbers in JSON files (0 = exact)")
    ap.add_argument("--ignore-key", action="append", default=[],
                    help="a JSON key that is not a result (wall-clock); repeatable")
    ap.add_argument("committed", type=pathlib.Path)
    ap.add_argument("rebuilt", type=pathlib.Path)
    args = ap.parse_args()

    a_bytes, b_bytes = args.committed.read_bytes(), args.rebuilt.read_bytes()
    if a_bytes == b_bytes:
        print("identical")
        return 0
    # Either name decides: the committed side is often a temp file (git show) with no suffix.
    if ".json" not in (args.committed.suffix, args.rebuilt.suffix):
        print("different (not JSON: compared byte for byte)")
        return 1
    out: list[str] = []
    differences(json.loads(a_bytes), json.loads(b_bytes), "", args.rel_tol,
                set(args.ignore_key), out)
    if out:
        print(f"different: {len(out)} value(s), e.g. " + "; ".join(out[:3]))
        return 1
    ignored = f", ignoring {', '.join(args.ignore_key)}" if args.ignore_key else ""
    print(f"equal within rel-tol {args.rel_tol:g}{ignored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
