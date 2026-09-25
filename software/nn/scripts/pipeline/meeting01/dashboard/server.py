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
        if self._ga_tailer is not None:
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

        self._last_poll = time.time()
        self._rebuild_snapshot()

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
            "last_poll": self._last_poll,
        }

    def _ga_summary(self) -> list[dict[str, Any]]:
        return [s.summary() for s in self._ga_states.values()]

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


@app.get("/api/ga/remote")
def api_ga_remote_status(results_dir: str = "results/meeting01", run_tag: str = "meeting01_loso"):
    """Check if a local remote-GA JSONL file exists and return its last entry."""
    remote_path = os.path.join(results_dir, f"{run_tag}_ga_remote.jsonl")
    if not os.path.isfile(remote_path):
        return JSONResponse({"available": False, "cells": []})
    # Read last line
    try:
        with open(remote_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return JSONResponse({"available": False, "cells": []})
            # Seek backwards to find last newline
            fh.seek(max(0, size - 10000))
            tail = fh.read().decode("utf-8", errors="replace")
            last_line = tail.rsplit("\n", 2)[-2]  # second-to-last (last may be partial)
            data = json.loads(last_line)
            data["available"] = True
            return JSONResponse(data)
    except (OSError, json.JSONDecodeError):
        return JSONResponse({"available": False, "cells": []})


@app.post("/api/ga/remote/collect")
def api_ga_remote_collect(
    results_dir: str = "results/meeting01",
    run_tag: str = "meeting01_loso",
):
    """Trigger a one-shot SSH collection of GA data from GridUnesp.

    Returns the collected data and appends it to the local JSONL file.
    Requires gridunesp_config.py credentials to be configured.
    """
    try:
        sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
        from gridunesp_config import load_config
        from collect_ga_local import collect_once as _collect_once
    except ImportError as exc:
        return JSONResponse({"error": f"Missing dependency: {exc}"}, status_code=500)

    try:
        cfg = load_config()
        data = _collect_once(cfg, results_dir, run_tag)
        if data is None:
            return JSONResponse({"error": "SSH collection returned no data"}, status_code=502)
        # Append to local JSONL
        remote_path = os.path.join(results_dir, f"{run_tag}_ga_remote.jsonl")
        with open(remote_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data) + "\n")
        return JSONResponse(data)
    except SystemExit:
        return JSONResponse({"error": "Credentials not configured"}, status_code=500)
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


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
        import re
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

@app.get("/api/stream")
async def api_stream(
    results_dir: str = "results/meeting01",
    run_tag: str = "meeting01_loso",
):
    """Server-Sent Events stream.  Sends one JSON frame per panel per tick.
    Named frames: session, fold_grid, active_configs, completed_configs,
    marginals, aggregation, events, ga_summary, heartbeat."""
    import asyncio
    import json as _json

    async def event_generator():
        w = _get_watcher(results_dir, run_tag)
        while True:
            snap = w.snapshot()
            # Send each panel as a named SSE event
            for panel in (
                "summary", "fold_grid", "active_configs", "completed_configs",
                "marginals", "aggregation", "events", "ga_summary",
            ):
                data = snap.get(panel)
                if data is not None:
                    yield f"event: {panel}\ndata: {_json.dumps(data)}\n\n"
            # Heartbeat with last poll timestamp
            yield f"event: heartbeat\ndata: {_json.dumps({'ts': snap.get('last_poll', 0)})}\n\n"
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
