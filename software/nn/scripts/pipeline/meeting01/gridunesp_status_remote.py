#!/usr/bin/env python3
"""gridunesp_status_remote.py — one JSON status line per interval, for
gridunesp_tui.py to consume over a single persistent SSH session.

Runs ON the GridUnesp login node (invoked by gridunesp_tui.py the same way
remote_monitor.sh invokes monitor.py --plain: one `ssh` call, this script's own
--interval loop inside it, not a local polling loop reconnecting repeatedly --
same Fail2Ban-avoidance reasoning as every other script in this directory, see
.wiki/Guides/GridUnesp-Deployment.md).

Stdlib + monitor.py's ingestion classes ONLY (EventTailer / SessionState /
scan_completed_folds / dataset_roster_from_profile / default_profile_path,
imported directly -- monitor.py only imports `rich` lazily inside its rendering
functions, never at module level, so importing it here for just the model
classes never requires rich to be installed on GridUnesp). This is deliberate:
the training-progress numbers below are computed by the EXACT SAME code
monitor.py's own dashboards use, not a second, parallel reimplementation that
could quietly drift from it.

Emits one JSON object per line to stdout and flushes immediately -- a proper
line-delimited stream a caller can read incrementally, not a document it has to
wait for in full. A per-cycle exception is caught and reported as
{"error": "..."} for that cycle only; the loop itself never dies from a
transient failure (matching monitor.py's own poll-loop error handling).

Usage (normally invoked BY gridunesp_tui.py, not run by hand):
  python3 scripts/pipeline/meeting01/gridunesp_status_remote.py [--interval 20]
    [--results-dir results/meeting01] [--run-tag meeting01_loso] [--profile PATH]
    [--datasets-root PATH] [--once]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor import (  # noqa: E402  (path insert above must run first)
    EventTailer,
    SessionState,
    dataset_roster_from_profile,
    default_profile_path,
    scan_completed_folds,
)

# (dataset dir name) -> (expected file count, extension). Verified 2026-09-24
# against upstream sources: fsdDataset/audioMNIST_8k counted from a complete
# local copy (Jakobovski/free-spoken-digit-dataset, soerenab/AudioMNIST); eegmmidb/
# siena counted from PhysioNet's own RECORDS manifest at
# https://physionet.org/files/<name>/1.0.0/RECORDS. A stale total here only makes
# the percentage wrong, not the presence check -- see check_dataset_progress.sh's
# header comment for the same table (kept in sync by hand; this script does not
# import that one since it is bash).
DATASET_TOTALS: dict[str, tuple[int, str]] = {
    "fsdDataset": (3000, "wav"),
    "audioMNIST_8k": (30000, "wav"),
    "eegmmidb": (1526, "edf"),
    "siena": (41, "edf"),
}
DATASET_ORDER = ["fsdDataset", "audioMNIST_8k", "eegmmidb", "siena"]

# ensure_datasets.sh's atomic-rename staging dirs (2026-09-24 fix -- see
# .wiki/Guides/GridUnesp-Deployment.md's Troubleshooting section): the FINAL path
# below only exists once fully complete, so while a clone/resample is in flight the
# final dir's own size stays at 0 -- probing the staging dir too is what gives a
# progress bar and rate DURING that phase instead of a dead 0% until it jumps to
# 100% at the rename. eegmmidb/siena have no staging dir (wget -N writes directly
# into the final path, already resumable/incremental on its own).
DATASET_STAGING: dict[str, str] = {
    "fsdDataset": ".fsdDataset_staging",
    "audioMNIST_8k": ".audioMNIST_8k_staging",
}


def _du_sb(path: str) -> int:
    """Raw byte size (`du -sb`), 0 if the path is absent or `du` fails -- 0 is
    indistinguishable from "genuinely empty," which is fine here: both mean
    "nothing to report a rate against yet."""
    try:
        out = subprocess.run(["du", "-sb", path], capture_output=True, text=True, timeout=30)
        return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout else 0
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return 0


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024.0:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}P"


def collect_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "ensure_datasets|wget|cmake --preset|cmake --build|srun|gridunesp_deploy"],
            capture_output=True, text=True, timeout=15,
        )
        return [ln for ln in out.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


class ProgressTracker:
    """Persists across loop iterations (one instance for the whole process's
    lifetime) so rate/ETA come from comparing THIS run's own consecutive samples --
    no blocking sleep-and-resample the way check_dataset_progress.sh's one-shot
    check needs; the natural --interval gap between cycles IS the sample window."""

    def __init__(self) -> None:
        self._prev: dict[str, tuple[float, int]] = {}

    def sample(self, name: str, ts: float, size_bytes: int, done: int, total: int) -> dict[str, Any]:
        rate_bytes_s: Optional[float] = None
        eta_s: Optional[float] = None
        prev = self._prev.get(name)
        if prev is not None:
            prev_ts, prev_bytes = prev
            dt = ts - prev_ts
            delta = size_bytes - prev_bytes
            # delta < 0 happens once, harmlessly, at the exact cycle a staging dir
            # (small/partial) gets replaced by the just-completed final dir jumping
            # straight to full size -- not a real negative rate, just don't report
            # a rate for that one cycle rather than show something absurd.
            if dt > 0 and delta >= 0:
                rate_bytes_s = delta / dt
        self._prev[name] = (ts, size_bytes)

        if rate_bytes_s and rate_bytes_s > 0 and 0 < done < total:
            avg_bytes_per_file = size_bytes / done
            remaining_bytes = avg_bytes_per_file * (total - done)
            eta_s = remaining_bytes / rate_bytes_s
        return {"rate_bytes_s": rate_bytes_s, "eta_s": eta_s}


def collect_datasets(datasets_root: str, tracker: ProgressTracker, ts: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in DATASET_ORDER:
        total, ext = DATASET_TOTALS[name]
        dir_path = os.path.join(datasets_root, name)
        present = os.path.isdir(dir_path)
        done = len(glob.glob(os.path.join(dir_path, "**", f"*.{ext}"), recursive=True)) if present else 0
        complete = done >= total

        final_bytes = _du_sb(dir_path) if present else 0
        staging_dir = DATASET_STAGING.get(name)
        staging_bytes = 0
        phase = "complete" if complete else "pending"
        if not complete and staging_dir:
            staging_path = os.path.join(datasets_root, staging_dir)
            if os.path.isdir(staging_path):
                staging_bytes = _du_sb(staging_path)
                phase = "staging"  # cloning / resampling, not yet renamed into place
        elif not complete and present:
            phase = "downloading"  # eegmmidb/siena: wget writes straight into place
        size_bytes = final_bytes if final_bytes else staging_bytes

        prog = tracker.sample(name, ts, size_bytes, done, total)
        out[name] = {
            "present": present,
            "done": done,
            "total": total,
            "complete": complete,
            "phase": phase,
            "size": _human_size(size_bytes) if size_bytes else "-",
            "rate_bytes_s": prog["rate_bytes_s"],
            "eta_s": prog["eta_s"],
        }
    return out


def collect_build(remote_dir: str) -> dict[str, bool]:
    cache = os.path.join(remote_dir, "out", "build", "max-performance", "CMakeCache.txt")
    binary = os.path.join(remote_dir, "out", "build", "max-performance",
                          "src", "experiments", "meeting01", "meeting01")
    return {"configured": os.path.isfile(cache), "built": os.path.isfile(binary)}


def collect_squeue() -> list[dict[str, str]]:
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--noheader",
             "--format=%i|%P|%j|%T|%M|%D|%R"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 7:
            continue
        jobid, partition, name, state, elapsed, nodes, reason = parts
        rows.append({"jobid": jobid, "partition": partition, "name": name, "state": state,
                     "elapsed": elapsed, "nodes": nodes, "reason": reason})
    return rows


def collect_training(results_dir: str, run_tag: str, profile: str | None) -> dict[str, Any]:
    tailer = EventTailer(results_dir, run_tag)
    state = SessionState()
    roster = dataset_roster_from_profile(profile or default_profile_path(run_tag))
    state.set_dataset_roster(roster)
    for ev in tailer.poll():
        state.apply(ev)
    state.reconcile(tailer.mtimes, tailer.missing)
    state.note_completed_folds(scan_completed_folds(results_dir, run_tag))

    if not state.session:
        return {"has_data": False}

    c = state.counts()
    elapsed = (time.time() - state.started_wall) if state.started_wall else None
    active = []
    for a in state.active_configs()[:6]:
        active.append({
            "where": f"{a.dataset} fold {a.fold}",
            "model": a.model,
            "encoding": a.encoding,
            "epoch": a.cur_progress_epoch or len(a.epochs),
            "max_epochs": a.max_epochs,
            "train_loss": a.last_train,
            "val_loss": a.last_val,
            "best_val": a.running_best_val,
        })
    failed = [f"{p.dataset} fold {p.fold}: {p.error}" for p in state.procs.values() if p.error]

    return {
        "has_data": True,
        "run_tag": state.session.get("run_tag"),
        "done": c["done"], "running": c["running"], "failed": c["failed"], "total": c["total"],
        "total_kind": "fixed" if roster else "estimated",
        "elapsed_s": elapsed,
        "eta_s": state.eta_seconds(),
        "active": active,
        "failed_procs": failed,
    }


def collect_once(remote_dir: str, results_dir: str, run_tag: str, profile: str | None,
                 datasets_root: str, tracker: ProgressTracker) -> dict[str, Any]:
    ts = time.time()
    return {
        "ts": ts,
        "processes": collect_processes(),
        "datasets": collect_datasets(datasets_root, tracker, ts),
        "build": collect_build(remote_dir),
        "squeue": collect_squeue(),
        "training": collect_training(results_dir, run_tag, profile),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--remote-dir", default=".", help="checkout root (cwd if run from there)")
    ap.add_argument("--results-dir", default="results/meeting01")
    ap.add_argument("--run-tag", default="meeting01_loso")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--datasets-root",
                    default=os.path.expanduser("~/Documentos/academico/UNESP/doutorado/databases"))
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    tracker = ProgressTracker()
    try:
        while True:
            try:
                blob = collect_once(args.remote_dir, args.results_dir, args.run_tag,
                                    args.profile, args.datasets_root, tracker)
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the stream
                blob = {"ts": time.time(), "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(blob), flush=True)
            if args.once:
                return 0
            time.sleep(max(1.0, args.interval))
    except (KeyboardInterrupt, BrokenPipeError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
