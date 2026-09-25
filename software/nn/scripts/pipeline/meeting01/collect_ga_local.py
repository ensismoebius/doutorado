#!/usr/bin/env python3
"""collect_ga_local.py — one-shot SSH collector for GA cache data from GridUnesp.

SSHes into GridUnesp, runs collect_ga from gridunesp_status_remote.py, and
saves the output as a local JSONL file the dashboard can tail.

Usage:
  python3 collect_ga_local.py [--run-tag meeting01_loso] [--output-dir results/meeting01]
                              [--interval 0]  # 0 = one-shot, >0 = continuous

Output file: <output_dir>/<run_tag>_ga_remote.jsonl
Each line: {"cells": [...], "ts": ..., "source": "remote"}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from gridunesp_config import load_config


def collect_once(cfg, results_dir: str, run_tag: str) -> dict | None:
    """SSH into GridUnesp and run the GA collection."""
    remote_results = os.path.join(cfg.remote_dir, results_dir)
    remote_cmd = (
        f"cd {cfg.remote_dir} && "
        f"source activate {cfg.conda_env} 2>/dev/null || conda activate {cfg.conda_env} 2>/dev/null; "
        f"python3 -c \""
        f"import sys; sys.path.insert(0, 'scripts/pipeline/meeting01'); "
        f"from gridunesp_status_remote import collect_ga; "
        f"import json; print(json.dumps(collect_ga('{remote_results}', '{run_tag}')))"
        f"\""
    )
    try:
        out = subprocess.run(
            cfg.ssh_cmd(remote_cmd),
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            print(f"[collect_ga] SSH error (rc={out.returncode}): {out.stderr.strip()[:200]}",
                  file=sys.stderr)
            return None
        data = json.loads(out.stdout.strip())
        data["ts"] = time.time()
        data["source"] = "remote"
        return data
    except subprocess.TimeoutExpired:
        print("[collect_ga] SSH timeout", file=sys.stderr)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[collect_ga] error: {exc}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-tag", default="meeting01_loso")
    ap.add_argument("--results-dir", default="results/meeting01")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write the local JSONL (default: results_dir)")
    ap.add_argument("--interval", type=float, default=0,
                    help="0 = one-shot; >0 = poll every N seconds")
    args = ap.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.run_tag}_ga_remote.jsonl")

    cfg = load_config()
    print(f"[collect_ga] target: {cfg.ssh_target}, output: {output_path}")

    while True:
        data = collect_once(cfg, args.results_dir, args.run_tag)
        if data:
            with open(output_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(data) + "\n")
            n_cells = len(data.get("cells", []))
            print(f"[collect_ga] {n_cells} cells written at {data['ts']:.0f}")
        else:
            print("[collect_ga] no data this cycle")

        if args.interval <= 0:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
