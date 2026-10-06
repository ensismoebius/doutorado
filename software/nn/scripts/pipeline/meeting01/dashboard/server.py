#!/usr/bin/env python3
"""server.py — FastAPI backend for the meeting01 web dashboard.

Read-only by construction: never writes into results/meeting01/ and never
signals any meeting01 process.  Closing/crashing the server is always safe;
reopening it just re-reads files from disk.

Usage
-----
    python dashboard/server.py --results-dir results/meeting01 --run-tag meeting01_loso

Endpoints
    GET /api/session          session summary (counts, ETA, config)
    GET /api/folds            fold × dataset grid
    GET /api/configs/active   currently training configs (with sweep_uncertain flag)
    GET /api/configs/completed  finished configs ranked by loss
    GET /api/configs/{id}/history  per-config epoch history
    GET /api/marginals        per-dimension best-score breakdown
    GET /api/aggregation      per (model, encoding) mean/std
    GET /api/events           recent event log
    GET /api/ga/summary       GA progress per active cell
    GET /api/ga/individuals   GA individual data per cell
    GET /api/runs             list historical runs found on disk
    GET /api/stream           SSE stream (polls every POLL_INTERVAL seconds)
    GET /api/health           health check
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import re
import sys
import time
from typing import Any, Optional

# Allow running from repo root or from the scripts directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SCRIPT_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from monitor import (
    _finite_or_none,
    EventTailer,
    SessionState,
    dataset_roster_from_profile,
    default_profile_path,
    scan_completed_folds,
)

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from ga_cache import GaCacheTailer, GaSearchState
except ImportError:
    GaCacheTailer = None  # type: ignore[assignment,misc]
    GaSearchState = None  # type: ignore[assignment,misc]

from sweep_state import annotate_sweep_uncertain

app = FastAPI(title="meeting01 dashboard", version="0.1.0")

# ── Global state ─────────────────────────────────────────────────────────────

POLL_INTERVAL = 2.0  # seconds between background polls

# Bounded tail window used when reading the last record of the remote-GA JSONL.
# One record is a few KB (per-cell summaries only), so 256 KiB is generous while
# keeping the read O(1) as the file grows under --interval polling.
_TAIL_WINDOW_BYTES = 256 * 1024

# How many *completed* GA searches keep a collapsible row in the GA tab.  Cache
# files are deleted on completion, so without a cap the summary list — and every
# SSE frame carrying it — would grow by one entry per finished cell for the
# whole run (480 cells under the standard profile).
MAX_GA_FINISHED = 64


class RunWatcher:
    """Owns one EventTailer + SessionState + GaCacheTailer pair and polls them
    in the background.  The snapshot is the latest JSON-serializable state,
    computed once per tick and served to every REST request."""

    def __init__(
        self,
        results_dir: str,
        run_tag: str,
        profile_path: Optional[str] = None,
    ) -> None:
        self.results_dir = results_dir
        self.run_tag = run_tag
        self._tailer = EventTailer(results_dir, run_tag)
        self._state = SessionState()
        self._ga_tailer: Optional[Any] = None
        self._ga_states: dict[tuple[str, int, int, str], Any] = {}
        # Collapsed summaries of searches whose cache file has disappeared.
        # Insertion-ordered; trimmed to MAX_GA_FINISHED in _collapse_finished_ga.
        self._ga_finished: dict[tuple[str, int, int, str], dict[str, Any]] = {}
        self._snapshot: dict[str, Any] = {}
        self._last_poll = 0.0
        self._consecutive_errors = 0

        # Initialize dataset roster
        pp = profile_path or default_profile_path(run_tag)
        roster = dataset_roster_from_profile(pp)
        self._state.set_dataset_roster(roster)

        # Initial drain
        self._do_poll()

        # Initialize GA tailer if available
        if GaCacheTailer is not None:
            self._ga_tailer = GaCacheTailer(results_dir, run_tag)

    def _do_poll(self) -> None:
        """One tailer.poll() → state.apply() round."""
        try:
            for ev in self._tailer.poll():
                self._state.apply(ev)
            self._state.reconcile(self._tailer.mtimes, self._tailer.missing)
            self._state.note_completed_folds(
                scan_completed_folds(self.results_dir, self.run_tag)
            )
            self._consecutive_errors = 0
        except Exception:
            self._consecutive_errors += 1
            if self._consecutive_errors > 30:
                raise

        # GA cache poll
        if self._ga_tailer is not None and GaSearchState is not None:
            for ind in self._ga_tailer.poll():
                key = (
                    ind.get("_dataset", ""),
                    ind.get("_fold", 0),
                    ind.get("_run_id", 0),
                    ind.get("_family", "snn"),
                )
                if key not in self._ga_states:
                    self._ga_states[key] = GaSearchState(*key)
                self._ga_states[key].add(ind)
            self._collapse_finished_ga()

        self._last_poll = time.time()
        self._rebuild_snapshot()

    def _collapse_finished_ga(self) -> None:
        """Drop per-individual state for GA searches whose cache file is gone.

        Cache files are deleted on successful completion
        (remove_checkpoint_artifacts), so a cell that stops appearing in
        active_cells() has finished.  Without this the watcher would retain
        every genome of every family for the whole run — 480 cells x ~40
        individuals for the standard profile — and _ga_summary() would rebuild
        and re-serialize all of it into every SSE frame, every 2 s, for weeks.

        The collapsed summary is kept (bounded to MAX_GA_FINISHED, most recent
        first) so a just-finished search stays visible instead of vanishing.
        """
        live = self._ga_tailer.active_cells() if self._ga_tailer is not None else set()
        for key in [k for k in self._ga_states if k not in live]:
            state = self._ga_states.pop(key)
            self._ga_finished[key] = state.summary()
        if len(self._ga_finished) > MAX_GA_FINISHED:
            # dict preserves insertion order — drop the oldest keys.
            for stale in list(self._ga_finished)[: len(self._ga_finished) - MAX_GA_FINISHED]:
                del self._ga_finished[stale]

    def _rebuild_snapshot(self) -> None:
        active = self._state.active_configs_json()
        if self._ga_tailer is not None:
            active = annotate_sweep_uncertain(active, self._ga_tailer.active_cells())
        self._snapshot = {
            "summary": self._state.session_summary_json(),
            "fold_grid": self._state.fold_grid_json(),
            "active_configs": active,
            "completed_configs": self._state.completed_configs_json(),
            "marginals": self._state.marginals_json(),
            "aggregation": self._state.aggregation_json(),
            "events": self._state.events_json(),
            "procs": self._state.procs_json(),
            "ga_summary": self._ga_summary(),
            "ga_finished_count": len(self._ga_finished),
            "last_poll": self._last_poll,
        }

    def _ga_summary(self) -> list[dict[str, Any]]:
        """In-flight searches first, then the most recent finished ones.

        Ordering matters: the GA tab is about watching searches progress, so
        live cells must not be pushed off the end of the list by a backlog of
        completed ones.
        """
        live = [s.summary() for s in self._ga_states.values()]
        return live + list(self._ga_finished.values())[:MAX_GA_FINISHED]

    def snapshot(self) -> dict[str, Any]:
        """Return the latest snapshot, polling if stale."""
        if time.time() - self._last_poll > POLL_INTERVAL:
            self._do_poll()
        return self._snapshot

    def config_history(self, config_id: str) -> Optional[dict[str, Any]]:
        """Return the full epoch history for one config."""
        cfg = self._state.configs.get(config_id)
        if cfg is None:
            return None
        return {
            "config_id": cfg.config_id,
            "dataset": cfg.dataset,
            "fold": cfg.fold,
            "model": cfg.model,
            "encoding": cfg.encoding,
            "status": cfg.status,
            "epochs": [{"epoch": e, "train": t, "val": v} for e, t, v in cfg.epochs],
        }


# ── Watcher registry ─────────────────────────────────────────────────────────

_watchers: dict[str, RunWatcher] = {}


def _get_watcher(results_dir: str, run_tag: str) -> RunWatcher:
    key = f"{results_dir}::{run_tag}"
    if key not in _watchers:
        _watchers[key] = RunWatcher(results_dir, run_tag)
    return _watchers[key]


# ── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/api/session")
def api_session(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    w = _get_watcher(results_dir, run_tag)
    snap = w.snapshot()
    return JSONResponse(snap.get("summary", {}))


@app.get("/api/folds")
def api_folds(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    w = _get_watcher(results_dir, run_tag)
    return JSONResponse(w.snapshot().get("fold_grid", []))


@app.get("/api/configs/active")
def api_active(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    w = _get_watcher(results_dir, run_tag)
    return JSONResponse(w.snapshot().get("active_configs", []))


@app.get("/api/configs/completed")
def api_completed(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    w = _get_watcher(results_dir, run_tag)
    return JSONResponse(w.snapshot().get("completed_configs", []))


@app.get("/api/configs/{config_id}/history")
def api_config_history(
    config_id: str,
    results_dir: str = "results/meeting01",
    run_tag: str = "meeting01_loso",
):
    w = _get_watcher(results_dir, run_tag)
    hist = w.config_history(config_id)
    if hist is None:
        return JSONResponse({"error": "config not found"}, status_code=404)
    return JSONResponse(hist)


@app.get("/api/marginals")
def api_marginals(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    w = _get_watcher(results_dir, run_tag)
    return JSONResponse(w.snapshot().get("marginals", {}))


@app.get("/api/aggregation")
def api_aggregation(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    w = _get_watcher(results_dir, run_tag)
    return JSONResponse(w.snapshot().get("aggregation", []))


@app.get("/api/events")
def api_events(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    w = _get_watcher(results_dir, run_tag)
    return JSONResponse(w.snapshot().get("events", []))


@app.get("/api/ga/summary")
def api_ga_summary(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    w = _get_watcher(results_dir, run_tag)
    return JSONResponse(w.snapshot().get("ga_summary", []))


@app.get("/api/ga/individuals")
def api_ga_individuals(
    results_dir: str = "results/meeting01",
    run_tag: str = "meeting01_loso",
    dataset: Optional[str] = None,
    fold: Optional[int] = None,
    family: Optional[str] = None,
):
    w = _get_watcher(results_dir, run_tag)
    rows = []
    for s in w._ga_states.values():
        if dataset and s.dataset != dataset:
            continue
        if fold is not None and s.fold != fold:
            continue
        if family and s.family != family:
            continue
        for ind in s.individuals:
            rows.append({
                "dataset": s.dataset,
                "fold": s.fold,
                "run_id": s.run_id,
                "family": s.family,
                **{k: v for k, v in ind.items() if not k.startswith("_")},
            })
    return JSONResponse(rows)


def _last_jsonl_record(path: str) -> Optional[dict[str, Any]]:
    """Return the last complete JSON object in a JSONL file, or None.

    Reads only a bounded tail window rather than the whole file (these grow
    without limit under --interval polling).  A trailing partial line — the
    normal state while collect_ga_local.py is mid-append — is skipped in
    favour of the last *parseable* record.

    A missing, empty or entirely unparseable file yields None rather than
    raising: "no remote snapshot yet" is the normal state before the first
    collection, not an error worth a 500.
    """
    try:
        with open(path, "rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            if size == 0:
                return None
            window = min(size, _TAIL_WINDOW_BYTES)
            fh.seek(size - window)
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None

    # Walk backwards over non-empty lines; the first one that parses wins.
    for line in reversed(tail.split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial/truncated tail record — try the previous one
        if isinstance(rec, dict):
            # Sanitize here rather than at each call site: json.loads accepts the
            # non-standard NaN/Infinity tokens by default, so a NaN can enter
            # through this parse and then make the eventual JSONResponse raise.
            return _finite_or_none(rec)
    return None


@app.get("/api/ga/remote")
def api_ga_remote_status(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    """Check if a local remote-GA JSONL file exists and return its last entry."""
    remote_path = os.path.join(results_dir, f"{run_tag}_ga_remote.jsonl")
    data = _last_jsonl_record(remote_path) if os.path.isfile(remote_path) else None
    if data is None:
        return JSONResponse({"available": False, "cells": []})
    data["available"] = True
    return JSONResponse(_finite_or_none(data))


@app.post("/api/ga/remote/collect")
def api_ga_remote_collect(
    results_dir: str = "results/meeting01",
    run_tag: str = "meeting01_loso",
):
    """Trigger a one-shot SSH collection of GA data from GridUnesp.

    Returns the collected data and appends it to the local JSONL file.
    Requires gridunesp_config.py credentials to be configured.

    Synchronous and potentially slow (one SSH round-trip plus a cold conda
    activate, bounded by collect_ga_local.SSH_TIMEOUT_S).  Run this against a
    single-worker server; under multiple uvicorn workers each request is handled
    by whichever worker is free, so a burst of clicks fans out into concurrent
    SSH sessions — which is exactly what GridUnesp's Fail2Ban lockout punishes.
    """
    try:
        from gridunesp_config import load_config
        from collect_ga_local import append_record, collect_once
    except ImportError as exc:
        return JSONResponse({"error": f"Missing dependency: {exc}"}, status_code=500)

    remote_path = os.path.join(results_dir, f"{run_tag}_ga_remote.jsonl")
    try:
        cfg = load_config()
    except SystemExit:
        return JSONResponse(
            {"error": "GridUnesp credentials not configured (set GRIDUNESP_USER "
                      "in scripts/pipeline/meeting01/.env, or use an SSH key)"},
            status_code=500,
        )

    data = collect_once(cfg, results_dir, run_tag)
    if data is None:
        return JSONResponse(
            {"error": "SSH collection failed — see the server log for the "
                      "ssh/conda error"},
            status_code=502,
        )
    try:
        append_record(remote_path, data)
    except OSError as exc:
        return JSONResponse(
            {"error": f"collected OK but could not write {remote_path}: {exc}"},
            status_code=500,
        )
    return JSONResponse(_finite_or_none(data))


@app.get("/api/runs")
def api_runs(results_dir: str = "results/meeting01"):
    """List historical runs found on disk by scanning for events files."""
    pattern = os.path.join(results_dir, "*_events.jsonl")
    run_tags: dict[str, dict[str, Any]] = {}
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        # Strip _events.jsonl suffix to get the run identifier
        identifier = basename.replace("_events.jsonl", "")
        # Extract run_tag (everything before the first _<dataset>_fold pattern)
        m = re.match(r"^(.+?)_([a-z0-9]+)_fold(\d+)$", identifier)
        if m:
            tag = m.group(1)
            if tag not in run_tags:
                run_tags[tag] = {"run_tag": tag, "files": 0, "datasets": set()}
            run_tags[tag]["files"] += 1
            run_tags[tag]["datasets"].add(m.group(2))

    runs = []
    for tag, info in sorted(run_tags.items()):
        runs.append({
            "run_tag": info["run_tag"],
            "events_files": info["files"],
            "datasets": sorted(info["datasets"]),
        })
    return JSONResponse(runs)


# ── SSE endpoint ─────────────────────────────────────────────────────────────

# Panels pushed as named SSE frames, in render order.
SSE_PANELS = (
    "summary", "fold_grid", "active_configs", "completed_configs",
    "marginals", "aggregation", "events", "ga_summary",
)


class SseDedupe:
    """Suppresses SSE frames whose payload has not changed since last sent.

    A dashboard tab left open for days would otherwise re-send every panel
    every POLL_INTERVAL — including `aggregation` and `completed_configs`,
    which can sit byte-identical for hours.  The heartbeat frame is emitted
    unconditionally, so the client still gets proof the stream is alive.
    """

    def __init__(self) -> None:
        self._last: dict[str, str] = {}

    def frames(self, snap: dict[str, Any]) -> list[str]:
        """Return the frames to send for this tick (empty when nothing moved)."""
        out: list[str] = []
        for panel in SSE_PANELS:
            data = snap.get(panel)
            if data is None:
                continue
            try:
                # allow_nan=False makes a stray NaN a caught error rather than
                # an invalid `data:` payload the browser silently drops.
                blob = json.dumps(_finite_or_none(data), allow_nan=False)
            except (TypeError, ValueError):
                continue  # one bad panel must not kill the stream
            if self._last.get(panel) == blob:
                continue
            self._last[panel] = blob
            out.append(f"event: {panel}\ndata: {blob}\n\n")
        return out


@app.get("/api/stream")
async def api_stream(
    results_dir: str = "results/meeting01",
    run_tag: str = "meeting01_loso",
):
    """Server-Sent Events stream.  Sends one JSON frame per changed panel per
    tick, plus an unconditional heartbeat.

    Named frames: summary, fold_grid, active_configs, completed_configs,
    marginals, aggregation, events, ga_summary, heartbeat."""
    import asyncio

    async def event_generator():
        w = _get_watcher(results_dir, run_tag)
        dedupe = SseDedupe()
        while True:
            snap = w.snapshot()
            for frame in dedupe.frames(snap):
                yield frame
            # Heartbeat carries last_poll so the client can show staleness.
            yield (f"event: heartbeat\ndata: "
                   f"{json.dumps({'ts': snap.get('last_poll', 0)})}\n\n")
            await asyncio.sleep(POLL_INTERVAL)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def api_health():
    return {"status": "ok"}


# ── Mount static files (must be last — catches /*) ──────────────────────────

_STATIC_DIR = os.path.join(_SCRIPT_DIR, "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="meeting01 web dashboard backend",
    )
    parser.add_argument(
        "--results-dir", default="results/meeting01",
        help="Path to results directory (default: results/meeting01)",
    )
    parser.add_argument(
        "--run-tag", default="meeting01_loso",
        help="Run tag to monitor (default: meeting01_loso)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--profile", default=None,
        help="Path to profile JSON (default: auto-detected from run_tag)",
    )
    args = parser.parse_args()

    # Validate results dir exists
    if not os.path.isdir(args.results_dir):
        print(f"Error: results directory not found: {args.results_dir}", file=sys.stderr)
        return 1

    # Pre-create the watcher so startup errors surface immediately
    _get_watcher(args.results_dir, args.run_tag)

    print(f"Serving dashboard at http://{args.host}:{args.port}")
    print(f"  results_dir = {os.path.abspath(args.results_dir)}")
    print(f"  run_tag     = {args.run_tag}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
