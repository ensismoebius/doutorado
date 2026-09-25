#!/usr/bin/env python3
"""monitor.py — live dashboard for the nested-LOSO meeting01 run.

The run is up to 18 independent `meeting01` processes (one per dataset x outer fold),
each appending structured events to

    results/meeting01/<run_tag>_<dataset>_fold<f>_events.jsonl

(schema v1, written by Meeting01Events.cpp / Meeting01EventCallback.hpp). This script
tails all of them, folds the events into one session model, and renders a compact
terminal dashboard.

What the panels mean
  SESSION        run identity + how many trainings are done / running / failed and a
                 rough ETA (done-so-far rate extrapolated over the whole grid).
  TRAINING NOW   the config(s) with a live epoch: which dataset/fold/model, epoch
                 N/M, current train & validation loss, the best validation loss seen
                 so far and at which epoch, the train/validation gap, how many
                 epochs since the last improvement, and a train/val loss sparkline.
  COMPLETED      every finished config ranked by loss (held-out *test* loss once the
                 config has been evaluated on the test speaker, otherwise the best
                 inner-validation loss). `monitor.py --rank N` prints one row's full
                 detail (all metrics + reproducibility).
  SEARCH SPACE   the hyperparameter grid, how far through it we are, and -- once
                 there is data -- the best inner-validation loss per SNN sweep
                 dimension (a descriptive summary, not a causal claim).
  RECENT         the last events, newest at the bottom.

OBSERVABILITY ONLY. Never writes to the run, never signals it; safe to start, kill,
and re-attach at any time. Ctrl-C to exit.

Modes
  (default)      live dashboard (needs `rich`, in scripts/requirements.txt)
  --plain        periodic plain-text snapshots; also automatic when stdout is not a
                 TTY (CI / redirect / `| tee`).  --once prints one snapshot.  Colored
                 automatically when the terminal supports it; a legend at the bottom
                 spells out every abbreviation (val, gap, ETA, mae, ...).
  --color        force ANSI colors in --plain output even when piped (`| less -R`).
  --no-color     disable ANSI colors in --plain output (also honors $NO_COLOR).
  --rank N       print the full detail of completed config #N (as ranked) and exit.
  --self-test    synthetic known-answer checks of the aggregator; stdlib only; CI-safe.

Descriptive only -- factual training diagnostics, no inferential or causal language
("overfitting", "converged", "significant", "generalizes").
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 1
_DATASET_RE = re.compile(r"_(?P<ds>[a-z0-9]+)_fold(?P<fold>\d+)_events\.jsonl$")
# A process is considered stalled/dead if its file has not grown in this long and it
# never emitted session_end.
STALE_SECONDS = 900.0


# --------------------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------------------
class EventTailer:
    """Follows every matching *_events.jsonl, yielding parsed event dicts in file order.

    Tracks a byte offset per file; on each poll it reads only appended bytes and parses
    whole lines (a trailing partial line is retried next poll). New files appearing in
    the directory are picked up automatically.
    """

    def __init__(self, results_dir: str, run_tag: str) -> None:
        self._pattern = os.path.join(results_dir, f"{glob.escape(run_tag)}*_fold*_events.jsonl")
        self._offsets: dict[str, int] = {}
        self._carry: dict[str, str] = {}
        self.mtimes: dict[str, float] = {}
        # Paths this poll() found gone that a PRIOR poll() had tracked — a fold's
        # events file removed out from under a live dashboard (a crashed attempt's
        # leftover cleaned up by hand, or a fresh RESUME=1 run skipping it because
        # the fold already has a complete CSV). Without this, a dashboard already
        # holding that path in memory has no way to learn the file is gone short of
        # ProcState.is_failed()'s STALE_SECONDS (15 min) timeout — it just keeps
        # showing a dead proc as "running" until that timer finally expires.
        self.missing: set[str] = set()

    def poll(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        current = set(glob.glob(self._pattern))
        self.missing = set(self._offsets) - current
        for path in self.missing:
            self._offsets.pop(path, None)
            self._carry.pop(path, None)
            self.mtimes.pop(path, None)
        for path in sorted(current):
            try:
                size = os.path.getsize(path)
                self.mtimes[path] = os.path.getmtime(path)
            except OSError:
                continue
            start = self._offsets.get(path, 0)
            if size < start:  # truncated (a fold re-run) -> restart from the top
                start = 0
                self._carry[path] = ""
            if size == start:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(start)
                    chunk = fh.read()
            except OSError:
                continue
            self._offsets[path] = size
            buf = self._carry.get(path, "") + chunk
            lines = buf.split("\n")
            self._carry[path] = lines.pop()  # trailing partial (or "")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict):
                    ev.setdefault("_path", path)
                    out.append(ev)
        return out


_FOLD_CSV_RE = re.compile(r"_(?P<ds>[a-z0-9]+)_fold(?P<fold>\d+)_comparative_metrics\.csv$")


def scan_completed_folds(results_dir: str, run_tag: str) -> dict[tuple[str, int], int]:
    """Trainings already finished in an EARLIER invocation of this run_tag, read
    straight from each fold's ``*_comparative_metrics.csv`` — written exactly
    once, at the very end of that fold's run (``write_experiment_outputs``), so
    its presence is proof of real completion, not a guess.

    Without this, RESUME=1 correctly skips re-running an already-complete fold
    (see 01_meeting01_run_loso.sh), but that fold's process is no longer live
    and never will be again this session — the dashboard would otherwise count
    it as 0 done forever, understating true progress by everything finished
    before this invocation started.
    """
    out: dict[tuple[str, int], int] = {}
    pattern = os.path.join(results_dir, f"{glob.escape(run_tag)}_*_fold*_comparative_metrics.csv")
    for path in glob.glob(pattern):
        m = _FOLD_CSV_RE.search(path)
        if not m:
            continue
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                # one training may write a val row and a test row — count the
                # distinct (model, encoding, hyperparams, run, seed) identity,
                # not raw rows, so it lines up with how "done" counts live configs
                trainings = {
                    (row.get("model"), row.get("encoding"), row.get("architecture"),
                     row.get("v_th"), row.get("alpha"), row.get("run"), row.get("seed"))
                    for row in csv.DictReader(fh)
                }
        except OSError:
            continue
        out[(m["ds"], int(m["fold"]))] = len(trainings)
    return out
def default_profile_path(run_tag: str) -> str:
    """Where 01_meeting01_run_loso.sh's own profile normally lives for a given
    run_tag: `src/experiments/meeting01/profiles/<run-tag-with-dashes>.json`
    (every profile in that directory follows this run_tag-with-underscores ->
    filename-with-dashes convention -- e.g. run_tag "meeting01_loso" <->
    "meeting01-loso.json"). Only a default; --profile overrides it."""
    return os.path.join("src", "experiments", "meeting01", "profiles",
                        f"{run_tag.replace('_', '-')}.json")


def dataset_roster_from_profile(profile_path: str) -> Optional[list[str]]:
    """The FULL, fixed dataset list `evaluation.datasets` a LOSO profile
    declares up front -- e.g. ["fsdd", "audiomnist", "eegmmidb", "siena"] -- read once
    from the static config file, not inferred from which datasets happen to
    have produced an event so far.

    Without this, SessionState.grid_size() can only count a dataset once the
    run script has actually STARTED it (a fold_begin/completed-CSV was seen
    for it), so the "of ~N total" denominator silently jumps by a whole
    dataset's worth of trainings (2700+) partway through a multi-day run, the
    moment 01_meeting01_run_loso.sh's outer loop reaches the next dataset --
    even though that dataset's full workload was always going to happen. The
    total looks unstable and the ETA looks untrustworthy, but nothing was
    actually wrong; the profile already knew the true count from the start.

    Returns None (never raises) if the file is missing, unreadable, not JSON,
    or has no non-empty `evaluation.datasets` list -- this is an optional
    stabilizer for an observability-only estimate, not a correctness path;
    grid_size() falls back to its old seen-so-far heuristic in that case.
    """
    try:
        with open(profile_path, encoding="utf-8") as fh:
            profile = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    datasets = profile.get("evaluation", {}).get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return None
    return [str(d) for d in datasets]


# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# session model
# --------------------------------------------------------------------------------------
@dataclass
class ConfigState:
    config_id: str
    dataset: str = ""
    fold: int = -1
    model: str = ""
    encoding: str = ""
    role: str = ""
    run_id: int = 0
    seed: int = 0
    hyperparams: dict[str, Any] = field(default_factory=dict)
    max_epochs: int = 0
    lr: Optional[float] = None
    param_count: Optional[int] = None
    macs: Optional[int] = None
    epochs: list[tuple[int, Optional[float], Optional[float]]] = field(default_factory=list)
    status: str = "running"  # running | done | failed
    stop_reason: str = ""
    epochs_run: int = 0
    best_val: Optional[float] = None
    best_epoch: Optional[int] = None
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)  # keyed by split
    last_ts: float = 0.0
    _epoch_ms: list[float] = field(default_factory=list)
    # within-epoch progress, only populated while a slow epoch is running (an
    # `epoch` event clears it). See Meeting01EventCallback::on_batch_end.
    cur_progress_epoch: int = 0
    cur_batch: int = 0
    cur_total_batches: int = 0
    cur_batch_frac: float = 0.0
    cur_batch_loss: Optional[float] = None
    cur_epoch_elapsed_s: Optional[float] = None
    cur_epoch_eta_s: Optional[float] = None

    def _clear_batch_progress(self) -> None:
        self.cur_progress_epoch = 0
        self.cur_batch = 0
        self.cur_total_batches = 0
        self.cur_batch_frac = 0.0
        self.cur_batch_loss = None
        self.cur_epoch_elapsed_s = None
        self.cur_epoch_eta_s = None

    # ---- derived, descriptive only --------------------------------------------------
    @property
    def last_train(self) -> Optional[float]:
        return self.epochs[-1][1] if self.epochs else None

    @property
    def last_val(self) -> Optional[float]:
        return self.epochs[-1][2] if self.epochs else None

    @property
    def running_best_val(self) -> Optional[float]:
        vals = [v for _, _, v in self.epochs if v is not None and math.isfinite(v)]
        return min(vals) if vals else self.best_val

    @property
    def running_best_epoch(self) -> Optional[int]:
        best, be = None, None
        for e, _, v in self.epochs:
            if v is not None and math.isfinite(v) and (best is None or v < best):
                best, be = v, e
        return be if be is not None else self.best_epoch

    @property
    def gap(self) -> Optional[float]:
        if self.last_val is None or self.last_train is None:
            return None
        return self.last_val - self.last_train

    @property
    def no_improve(self) -> int:
        be = self.running_best_epoch
        if be is None or not self.epochs:
            return 0
        return max(0, self.epochs[-1][0] - be)

    @property
    def val_increased_epochs(self) -> int:
        """Consecutive most-recent epochs whose val loss rose vs the epoch before."""
        vs = [v for _, _, v in self.epochs if v is not None and math.isfinite(v)]
        n = 0
        for i in range(len(vs) - 1, 0, -1):
            if vs[i] > vs[i - 1]:
                n += 1
            else:
                break
        return n

    @property
    def rank_val(self) -> Optional[float]:
        for split in ("test", "val"):
            m = self.metrics.get(split)
            if m and m.get("mse") is not None:
                return m["mse"]
        return self.running_best_val

    @property
    def rank_val_kind(self) -> str:
        for split in ("test", "val"):
            m = self.metrics.get(split)
            if m and m.get("mse") is not None:
                return "test" if split == "test" else "val"
        return "best"

    def avg_epoch_ms(self) -> Optional[float]:
        return sum(self._epoch_ms) / len(self._epoch_ms) if self._epoch_ms else None


@dataclass
class ProcState:
    path: str
    dataset: str = ""
    fold: int = -1
    started: bool = False
    ended: bool = False
    error: str = ""
    last_ts: float = 0.0

    def is_failed(self, now: float, file_mtime: float) -> bool:
        if self.error:
            return True
        if self.ended:
            return False
        return self.started and (now - file_mtime) > STALE_SECONDS


@dataclass
class FoldCell:
    """One (dataset, fold)'s progress for the fold x dataset overview panel."""
    done: int = 0
    running: int = 0
    failed: int = 0
    total: int = 0

    @property
    def frac(self) -> float:
        """Fraction of this fold's expected trainings that are done, 0.0-1.0."""
        return self.done / self.total if self.total else 0.0

    @property
    def status(self) -> str:
        """One headline state to color the cell by: 'failed' (any failure seen,
        regardless of how much else finished), 'done' (every expected training
        finished), 'running' (some progress but not finished), or 'unstarted'."""
        if self.failed:
            return "failed"
        if self.total and self.done >= self.total:
            return "done"
        if self.running or self.done:
            return "running"
        return "unstarted"
class SessionState:
    def __init__(self) -> None:
        self.session: dict[str, Any] = {}
        self.configs: dict[str, ConfigState] = {}
        self.procs: dict[str, ProcState] = {}
        self.folds_seen: set[tuple[str, int]] = set()
        self.events: deque[dict[str, Any]] = deque(maxlen=400)
        self.selected: dict[str, dict[str, Any]] = {}
        self.started_wall: Optional[float] = None
        self.loop_error: str = ""  # last transient poll/render error, "" when clear
        # (dataset, fold) -> training count, from scan_completed_folds() — folds
        # that finished before THIS session started and so left no live config
        # behind to count individually. See note_completed_folds().
        self.completed_fold_trainings: dict[tuple[str, int], int] = {}
        # The FULL dataset roster from the run's own profile (see
        # dataset_roster_from_profile()), set once via set_dataset_roster().
        # None until set -- grid_size() then falls back to counting only the
        # datasets observed so far, which understates the true total until
        # every dataset has actually started.
        self.dataset_roster: Optional[list[str]] = None

    def note_completed_folds(self, mapping: dict[tuple[str, int], int]) -> None:
        """Merge in scan_completed_folds() results. A fold this session has
        itself observed live (a fold_begin was seen for it) is excluded even
        if also present here — its configs are already counted individually
        through self.configs as they finish, crediting the fold again from
        its CSV the moment it completes would double it."""
        self.completed_fold_trainings = {
            k: v for k, v in mapping.items() if k not in self.folds_seen
        }

    def set_dataset_roster(self, datasets: Optional[list[str]]) -> None:
        """Record the run's FULL, fixed dataset list (from
        dataset_roster_from_profile()) so grid_size() has a stable, known-
        upfront denominator instead of growing by a whole dataset's worth of
        trainings each time 01_meeting01_run_loso.sh's outer loop reaches the
        next one. A no-op for an empty/None list -- grid_size() keeps its
        seen-so-far fallback."""
        if datasets:
            self.dataset_roster = list(datasets)

    # ---- ingest -------------------------------------------------------------------
    def apply(self, ev: dict[str, Any]) -> None:
        if ev.get("v") != SCHEMA_VERSION:
            return
        etype = ev.get("type")
        ts = float(ev.get("ts_unix", 0.0) or 0.0)
        path = ev.get("_path", "")
        ds = ev.get("dataset", "")
        fold = int(ev.get("fold", -1))
        proc = self.procs.setdefault(path, ProcState(path=path, dataset=ds, fold=fold))
        proc.last_ts = max(proc.last_ts, ts)

        if etype == "session_begin":
            if not self.session:
                self.session = {k: v for k, v in ev.items() if not k.startswith("_")}
            self.started_wall = ts if self.started_wall is None else min(self.started_wall, ts)
            proc.started = True
        elif etype == "session_end":
            proc.ended = True
        elif etype == "session_error":
            proc.error = str(ev.get("what", "error"))
        elif etype == "fold_begin":
            self.folds_seen.add((ds, fold))
        elif etype == "config_begin":
            c = self._config(ev, ds, fold)
            c.model = ev.get("model", c.model)
            c.encoding = ev.get("encoding", c.encoding)
            c.role = ev.get("role", c.role)
            c.run_id = int(ev.get("run_id", c.run_id))
            c.seed = int(ev.get("seed", c.seed))
            c.hyperparams = ev.get("hyperparams", c.hyperparams) or c.hyperparams
            c.max_epochs = int(ev.get("max_epochs", c.max_epochs) or 0)
            c.lr = ev.get("lr", c.lr)
            c.param_count = ev.get("param_count") or c.param_count
            c.macs = ev.get("macs") or c.macs
            c.status = "running"
            c._clear_batch_progress()
            c.last_ts = ts
        elif etype == "epoch_progress":
            c = self._config(ev, ds, fold)
            c.cur_progress_epoch = int(ev.get("epoch", 0))
            c.cur_batch = int(ev.get("batch", 0))
            c.cur_total_batches = int(ev.get("total_batches", 0))
            c.cur_batch_frac = float(ev.get("frac", 0.0) or 0.0)
            c.cur_batch_loss = ev.get("batch_loss")
            c.cur_epoch_elapsed_s = ev.get("epoch_elapsed_s")
            c.cur_epoch_eta_s = ev.get("epoch_eta_s")
            c.max_epochs = int(ev.get("max_epochs", c.max_epochs) or c.max_epochs)
            c.last_ts = ts
        elif etype == "epoch":
            c = self._config(ev, ds, fold)
            c.epochs.append((int(ev.get("epoch", 0)), ev.get("train_loss"), ev.get("val_loss")))
            c.max_epochs = int(ev.get("max_epochs", c.max_epochs) or c.max_epochs)
            ems = ev.get("epoch_ms")
            if isinstance(ems, (int, float)) and math.isfinite(ems):
                c._epoch_ms.append(float(ems))
            c._clear_batch_progress()
            c.last_ts = ts
        elif etype == "train_end":
            c = self._config(ev, ds, fold)
            c.epochs_run = int(ev.get("epochs_run", len(c.epochs)))
            c.stop_reason = ev.get("stop_reason", c.stop_reason)
            c.best_val = ev.get("best_val_loss", c.best_val)
            c.best_epoch = ev.get("best_val_epoch", c.best_epoch)
            c.last_ts = ts
        elif etype == "config_end":
            c = self._config(ev, ds, fold)
            c.model = ev.get("model", c.model)
            c.encoding = ev.get("encoding", c.encoding)
            c.role = ev.get("role", c.role)
            c.run_id = int(ev.get("run_id", c.run_id))
            c.seed = int(ev.get("seed", c.seed))
            if ev.get("hyperparams"):
                c.hyperparams = ev["hyperparams"]
            split = ev.get("split", "val")
            c.metrics[split] = ev.get("metrics", {}) or {}
            if c.metrics[split].get("param_count"):
                c.param_count = c.metrics[split]["param_count"]
            if c.metrics[split].get("macs"):
                c.macs = c.metrics[split]["macs"]
            c.status = "done"
            c.last_ts = ts
        elif etype == "config_selected":
            self.selected[ev.get("config_id", "")] = ev.get("selected", {})

        if etype == "epoch_progress":
            return  # transient heartbeat — folded into ConfigState, never logged
        if etype != "epoch" or not self.events or self.events[-1].get("type") != "epoch" \
                or self.events[-1].get("config_id") != ev.get("config_id"):
            self.events.append({k: v for k, v in ev.items() if not k.startswith("_")})
        else:
            self.events[-1] = {k: v for k, v in ev.items() if not k.startswith("_")}

    def _config(self, ev: dict[str, Any], ds: str, fold: int) -> ConfigState:
        cid = ev.get("config_id", "?")
        c = self.configs.get(cid)
        if c is None:
            c = ConfigState(config_id=cid, dataset=ds, fold=fold)
            self.configs[cid] = c
        return c

    def reconcile(self, tailer_mtimes: dict[str, float], missing: "set[str] | None" = None) -> None:
        now = time.time()
        self._mtimes = dict(tailer_mtimes)
        for path, proc in self.procs.items():
            if missing and path in missing:
                # The events file itself is gone (RESUME=1 skipping an already-
                # complete fold, or a crashed attempt's leftover cleaned up by
                # hand) — that is unambiguous, immediate proof the proc is dead.
                # Do not make the dashboard wait out STALE_SECONDS to learn what
                # the filesystem already told it this poll.
                proc.error = proc.error or "events file removed (fold already complete, or run abandoned)"
                for c in self.configs.values():
                    if c.dataset == proc.dataset and c.fold == proc.fold and c.status == "running":
                        c.status = "failed"
                continue
            mt = tailer_mtimes.get(path, proc.last_ts)
            if proc.is_failed(now, mt):
                proc.error = proc.error or "process stopped without session_end"
                for c in self.configs.values():
                    if c.dataset == proc.dataset and c.fold == proc.fold and c.status == "running":
                        c.status = "failed"

    def live_procs(self) -> list["ProcState"]:
        """Procs whose events file was touched within STALE_SECONDS and that have
        not ended or errored — i.e. a training process is demonstrably alive even
        if it is momentarily between two configs (the fast SNN sweep does this)."""
        now = time.time()
        out = []
        for path, proc in self.procs.items():
            mt = getattr(self, "_mtimes", {}).get(path, proc.last_ts)
            if proc.started and not proc.ended and not proc.error \
                    and (now - mt) <= STALE_SECONDS:
                out.append(proc)
        return out

    # ---- derived views ----------------------------------------------------------
    def per_fold_trainings(self) -> int:
        """Matches Meeting01Experiment.cpp's total_outer_runs formula for one
        (dataset, fold) process: baselines still multiply by encodings (their
        training loop sweeps encoding as before); the SNN arm does not, since
        encoding/architecture are now NSGA-II genes rather than an outer sweep."""
        s = self.session.get("search_space", {})
        repeats = int(self.session.get("repeats", 1) or 1)
        n_encodings = len(s.get("encodings", []) or [1])
        baseline_runs = len(s.get("baselines", []) or []) * n_encodings * repeats
        has_snn = bool(s.get("snn_architectures"))
        ga_pop = int(s.get("ga_population_size", 0) or 0)
        ga_gen = int(s.get("ga_generations", 0) or 0)
        snn_runs = (ga_pop * (1 + ga_gen) * repeats) if has_snn else 0
        return baseline_runs + snn_runs

    def grid_size(self) -> int:
        n_fold = int(self.session.get("cv_num_folds", 1) or 1)
        if self.dataset_roster:
            # the run's own profile already declares the full dataset list --
            # a fixed, known-upfront denominator that never jumps mid-run as
            # 01_meeting01_run_loso.sh's outer loop reaches the next dataset
            n_ds = len(self.dataset_roster)
        else:
            n_ds = len(self.session.get("all_datasets", []) or [1])
            if n_ds == 1 and n_fold > 1:
                # a fold "seen" either live this session or as an already-complete
                # CSV from an earlier invocation (RESUME=1) both count towards
                # which datasets the grid actually spans -- an underestimate
                # until dataset_roster is available, but the least-wrong guess
                datasets = {d for d, _ in self.folds_seen} | {d for d, _ in self.completed_fold_trainings}
                n_ds = max(1, len(datasets))
        return self.per_fold_trainings() * n_fold * n_ds

    def fold_grid(self) -> dict[tuple[str, int], FoldCell]:
        """Per (dataset, fold) rollup of done/running/failed trainings out of
        that fold's expected total -- feeds the fold x dataset overview panel,
        the one place the dashboard shows the WHOLE nested-LOSO sweep's state
        instead of just whichever fold happens to be training right now.
        Combines this session's live configs with the same scan_completed_folds()
        credit counts() uses, so the two panels never disagree."""
        total = self.per_fold_trainings()
        grid: dict[tuple[str, int], FoldCell] = {}
        for cfg in self.configs.values():
            cell = grid.setdefault((cfg.dataset, cfg.fold), FoldCell(total=total))
            if cfg.status == "done":
                cell.done += 1
            elif cfg.status == "running":
                cell.running += 1
            elif cfg.status == "failed":
                cell.failed += 1
        for key, n_done in self.completed_fold_trainings.items():
            cell = grid.setdefault(key, FoldCell(total=total))
            cell.done += n_done
        return grid

    def counts(self) -> dict[str, int]:
        running = done = failed = 0
        for c in self.configs.values():
            if c.status == "running":
                running += 1
            elif c.status == "done":
                done += 1
            elif c.status == "failed":
                failed += 1
        done += sum(self.completed_fold_trainings.values())
        return {"running": running, "done": done, "failed": failed, "total": self.grid_size()}

    def eta_seconds(self) -> Optional[float]:
        if not self.started_wall:
            return None
        c = self.counts()
        elapsed = time.time() - self.started_wall
        if c["done"] < 1 or elapsed <= 0 or c["total"] <= 0:
            return None
        rate = c["done"] / elapsed
        return (c["total"] - c["done"]) / rate if rate > 0 else None

    def completed_configs(self) -> list[ConfigState]:
        return sorted(
            (c for c in self.configs.values()
             if c.status == "done" and c.role != "snn_sweep"),
            key=lambda c: (c.rank_val is None, c.rank_val if c.rank_val is not None else 0.0),
        )

    def active_configs(self) -> list[ConfigState]:
        return sorted(
            (c for c in self.configs.values()
             if c.status == "running" and (c.epochs or c.cur_total_batches > 0)),
            key=lambda c: -c.last_ts,
        )

    def marginals(self) -> dict[str, list[tuple[str, Optional[float], int]]]:
        dims: dict[str, dict[str, list[float]]] = {"architecture": {}, "v_th": {}, "alpha": {}}
        for c in self.configs.values():
            if c.role not in ("snn_sweep", "snn_final") or c.status != "done":
                continue
            score = c.rank_val
            if score is None:
                continue
            for key in dims:
                val = c.hyperparams.get(key)
                if val is not None:
                    dims[key].setdefault(str(val), []).append(score)
        return {
            key: [(k, min(v) if v else None, len(v)) for k, v in sorted(buckets.items())]
            for key, buckets in dims.items()
        }

    def aggregation(self) -> list[tuple[str, str, Optional[float], Optional[float], int, int]]:
        groups: dict[tuple[str, str], list[float]] = {}
        failed: dict[tuple[str, str], int] = {}
        for c in self.configs.values():
            if c.role == "snn_sweep":
                continue
            key = (c.model, c.encoding)
            if c.status == "failed":
                failed[key] = failed.get(key, 0) + 1
                continue
            if c.status != "done":
                continue
            v = c.rank_val
            if v is not None and math.isfinite(v):
                groups.setdefault(key, []).append(v)
        rows = []
        for key in sorted(set(groups) | set(failed)):
            vals = groups.get(key, [])
            mean = sum(vals) / len(vals) if vals else None
            std = (
                math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
                if mean is not None and len(vals) >= 2 else None
            )
            rows.append((key[0], key[1], mean, std, len(vals), failed.get(key, 0)))
        return rows


# --------------------------------------------------------------------------------------
# plain-JSON accessors (dashboard backend)
# --------------------------------------------------------------------------------------
# Every float passes through _finite_or_none before it reaches json.dumps —
# json.dumps(float('nan')) emits the literal token NaN, which is not valid
# JSON and silently breaks JSON.parse on the frontend the first time a run
# produces an Inf or NaN loss.
def _finite_or_none(v: Any) -> Any:
    """Return *v* unchanged unless it is a float that is NaN or Inf, in which
    case return None.  Applied recursively to nested lists/dicts."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite_or_none(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite_or_none(x) for x in v]
    return v


def _config_state_to_dict(c: ConfigState) -> dict[str, Any]:
    return _finite_or_none({
        "config_id": c.config_id,
        "dataset": c.dataset,
        "fold": c.fold,
        "model": c.model,
        "encoding": c.encoding,
        "role": c.role,
        "run_id": c.run_id,
        "seed": c.seed,
        "hyperparams": dict(c.hyperparams),
        "max_epochs": c.max_epochs,
        "lr": c.lr,
        "param_count": c.param_count,
        "macs": c.macs,
        "epochs": [{"epoch": e, "train": t, "val": v} for e, t, v in c.epochs],
        "status": c.status,
        "stop_reason": c.stop_reason,
        "epochs_run": c.epochs_run,
        "best_val": c.best_val,
        "best_epoch": c.best_epoch,
        "metrics": c.metrics,
        "last_train": c.last_train,
        "last_val": c.last_val,
        "gap": c.gap,
        "no_improve": c.no_improve,
        "rank_val": c.rank_val,
        "rank_val_kind": c.rank_val_kind,
        "cur_progress_epoch": c.cur_progress_epoch,
        "cur_batch": c.cur_batch,
        "cur_total_batches": c.cur_total_batches,
        "cur_batch_frac": c.cur_batch_frac,
        "cur_batch_loss": c.cur_batch_loss,
        "cur_epoch_elapsed_s": c.cur_epoch_elapsed_s,
        "cur_epoch_eta_s": c.cur_epoch_eta_s,
    })


def _proc_state_to_dict(p: ProcState) -> dict[str, Any]:
    return {
        "path": p.path,
        "dataset": p.dataset,
        "fold": p.fold,
        "started": p.started,
        "ended": p.ended,
        "error": p.error,
    }


def _fold_cell_to_dict(ds: str, fold: int, cell: FoldCell) -> dict[str, Any]:
    return {
        "dataset": ds,
        "fold": fold,
        "done": cell.done,
        "running": cell.running,
        "failed": cell.failed,
        "total": cell.total,
        "frac": cell.frac,
        "status": cell.status,
    }


# Methods added to SessionState for JSON-serializable snapshots consumed by the
# web dashboard backend.  Zero edits to any existing render_* / _panel_* function.
def _session_summary(self) -> dict[str, Any]:
    c = self.counts()
    return _finite_or_none({
        "session": dict(self.session),
        "counts": c,
        "eta_seconds": self.eta_seconds(),
        "started_wall": self.started_wall,
        "per_fold_trainings": self.per_fold_trainings(),
        "grid_size": self.grid_size(),
    })


def _fold_grid_json(self) -> list[dict[str, Any]]:
    return [
        _fold_cell_to_dict(ds, fold, cell)
        for (ds, fold), cell in sorted(self.fold_grid().items())
    ]


def _active_configs_json(self) -> list[dict[str, Any]]:
    return [_config_state_to_dict(c) for c in self.active_configs()]


def _completed_configs_json(self) -> list[dict[str, Any]]:
    return [_config_state_to_dict(c) for c in self.completed_configs()]


def _marginals_json(self) -> dict[str, list[dict[str, Any]]]:
    return {
        dim: [{"value": k, "best": _finite_or_none(b), "count": n}
              for k, b, n in entries]
        for dim, entries in self.marginals().items()
    }


def _aggregation_json(self) -> list[dict[str, Any]]:
    return [
        _finite_or_none({
            "model": model, "encoding": enc,
            "mean": mean, "std": std,
            "n_done": n_done, "n_failed": n_failed,
        })
        for model, enc, mean, std, n_done, n_failed in self.aggregation()
    ]


def _events_json(self) -> list[dict[str, Any]]:
    return [dict(ev) for ev in self.events]


def _procs_json(self) -> list[dict[str, Any]]:
    return [_proc_state_to_dict(p) for p in self.procs.values()]


def _full_snapshot(self) -> dict[str, Any]:
    return _finite_or_none({
        "summary": _session_summary(self),
        "fold_grid": _fold_grid_json(self),
        "active_configs": _active_configs_json(self),
        "completed_configs": _completed_configs_json(self),
        "marginals": _marginals_json(self),
        "aggregation": _aggregation_json(self),
        "events": _events_json(self),
        "procs": _procs_json(self),
    })


# Bind as methods on SessionState (kept as free functions above for testability
# and to avoid cluttering the class with closures over module-level helpers).
SessionState.session_summary_json = _session_summary  # type: ignore[attr-defined]
SessionState.fold_grid_json = _fold_grid_json  # type: ignore[attr-defined]
SessionState.active_configs_json = _active_configs_json  # type: ignore[attr-defined]
SessionState.completed_configs_json = _completed_configs_json  # type: ignore[attr-defined]
SessionState.marginals_json = _marginals_json  # type: ignore[attr-defined]
SessionState.aggregation_json = _aggregation_json  # type: ignore[attr-defined]
SessionState.events_json = _events_json  # type: ignore[attr-defined]
SessionState.procs_json = _procs_json  # type: ignore[attr-defined]
SessionState.full_snapshot_json = _full_snapshot  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# color (ANSI) -- the --plain renderer only; the --rich dashboard (default, TTY) has
# its own styling via `rich`. Off by default whenever stdout is not a terminal (a
# redirect, `| tee`, CI) so a log file never fills up with escape codes; --color
# forces it on (e.g. piping through `less -R`), --no-color forces it off, and the
# NO_COLOR env var (https://no-color.org) is honored the same as --no-color.
# --------------------------------------------------------------------------------------
_ANSI = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
         "blue": "34", "magenta": "35", "cyan": "36", "white": "37"}
_COLOR_ENABLED = sys.stdout.isatty()


def _set_color(enabled: bool) -> None:
    global _COLOR_ENABLED
    _COLOR_ENABLED = enabled


def _c(text: str, *styles: str) -> str:
    """Wrap `text` in ANSI SGR codes, or return it verbatim when color is off."""
    if not _COLOR_ENABLED or not text:
        return text
    return f"\033[{';'.join(_ANSI[s] for s in styles)}m{text}\033[0m"


_EVENT_COLOR = {"session_error": ("red", "bold"), "config_end": ("green",),
                "config_selected": ("yellow",), "fold_begin": ("cyan",),
                "fold_end": ("cyan",), "train_end": ("green",),
                "session_begin": ("magenta",)}


def _legend() -> list[str]:
    """One block of plain-English definitions for every abbreviation the dashboard
    uses, so a reader never has to guess what a column or field means."""
    return [
        "legend",
        "  val / train        = validation / training loss for the current epoch",
        "  gap(val-train)      = val loss minus train loss, last epoch (a spread, not a diagnosis)",
        "  no improvement N ep = epochs since the best validation loss so far",
        "  best val @ ep K     = lowest validation loss seen, and which epoch produced it",
        "  loss (COMPLETED)    = held-out TEST loss once evaluated, else the best inner-",
        "                        validation loss reached during the sweep (see the kind column",
        "                        in --rank / the rich dashboard)",
        "  mae                 = mean absolute error, same split as the loss column",
        "  ETA                 = trainings-remaining / (done-so-far / elapsed) -- a rough",
        "                        linear projection, not a guarantee",
        "  ~N (fixed)          = grid size read from the run's own profile -- stable for the",
        "                        whole run, does not shift as new datasets start",
        "  ~N (estimated)      = no profile found -- a guess from datasets seen live so far,",
        "                        which grows in jumps as each new dataset begins (untrustworthy",
        "                        ETA until every dataset has started; pass --profile to fix it)",
    ]
# --------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------
def _f(x: Optional[float], nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def _hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or not math.isfinite(seconds):
        return "--:--:--"
    s = int(seconds)
    # Past a day, hours-only gets unreadable (163:04:49) — lead with whole days.
    if s >= 86400:
        return f"{s // 86400}d {(s % 86400) // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _hp_str(hp: dict[str, Any]) -> str:
    """Table-cell form: a dash when there are no hyperparameters (baselines)."""
    return _hp_inline(hp) or "-"


def _hp_inline(hp: dict[str, Any]) -> str:
    """Prose form: empty string when there are no hyperparameters."""
    if not hp:
        return ""
    bits = []
    if "architecture" in hp:
        bits.append(str(hp["architecture"]))
    if "v_th" in hp:
        bits.append(f"v={float(hp['v_th']):.2f}")
    if "alpha" in hp:
        bits.append(f"a={float(hp['alpha']):.2f}")
    return " ".join(bits)


def _spark(values: list[Optional[float]], width: int = 40) -> str:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return "·" * len(vals)
    blocks = "▁▂▃▄▅▆▇█"
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[min(len(vals) - 1, int(i * step))] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    return "".join(blocks[min(7, int((v - lo) / rng * 7))] for v in vals)


def _bar(frac: float, width: int = 22) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(round(frac * width))
    return "█" * full + "░" * (width - full)


def _ascii_plot(values: list[Optional[float]], width: int = 56, height: int = 8) -> list[str]:
    """A filled area chart (Unicode eighth-blocks, one full column of shading
    per point) with the min/max labelled on the bottom/top row.

    Deliberately NOT one '*' per column at its own row: at a small height
    (the dashboard's 4-row TRAINING NOW chart) that scatter renders as
    disconnected dots with mostly-blank rows between them -- unreadable. A
    filled column reads as a shape at any height, 4 rows or 8, because the
    column *below* each value is shaded too, not just the single row nearest
    the value's height.
    """
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return ["(not enough points yet)"]
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[min(len(vals) - 1, int(i * step))] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    blocks = " ▁▂▃▄▅▆▇█"
    units = height * 8
    filled_units = [max(0, min(units, round((v - lo) / rng * units))) for v in vals]
    rows = []
    for row in range(height):
        row_from_bottom = height - 1 - row
        floor = row_from_bottom * 8
        rows.append("".join(blocks[max(0, min(8, u - floor))] for u in filled_units))
    pad = len(f"{hi:.4f} ")
    rows[0] = f"{hi:.4f} " + rows[0]
    for i in range(1, height - 1):
        rows[i] = " " * pad + rows[i]
    rows[-1] = f"{lo:.4f} ".rjust(pad) + rows[-1]
    return rows


def _event_line(ev: dict[str, Any]) -> tuple[str, str, str]:
    """(HH:MM:SS, TYPE, human summary) for one event."""
    t = time.strftime("%H:%M:%S", time.localtime(float(ev.get("ts_unix", 0))))
    et = ev.get("type", "?")
    if et == "epoch":
        s = (f"{ev.get('config_id', '')}  ep {ev.get('epoch')}/{ev.get('max_epochs')}  "
             f"train {_f(ev.get('train_loss'), 5)}  val {_f(ev.get('val_loss'), 5)}")
    elif et == "epoch_progress":
        s = (f"{ev.get('config_id', '')}  ep {ev.get('epoch')} batch "
             f"{ev.get('batch')}/{ev.get('total_batches')}")
    elif et == "config_begin":
        hp = _hp_inline(ev.get("hyperparams", {}))
        s = (f"{ev.get('model', '')}/{ev.get('encoding', '')}{('  ' + hp) if hp else ''}  "
             f"started ({ev.get('role', '')}, <= {ev.get('max_epochs', '?')} ep)")
    elif et == "train_end":
        s = (f"{ev.get('config_id', '')}  {ev.get('epochs_run', '?')} ep, "
             f"{ev.get('stop_reason', '')}, best val {_f(ev.get('best_val_loss'), 5)} "
             f"@ ep {ev.get('best_val_epoch', '?')}")
    elif et == "config_end":
        mm = ev.get("metrics", {}) or {}
        s = (f"{ev.get('config_id', '')}  [{ev.get('split')}]  "
             f"mse {_f(mm.get('mse'), 5)}  mae {_f(mm.get('mae'), 5)}")
    elif et == "config_selected":
        sel = ev.get("selected", {})
        s = f"SNN winner: {_hp_str(sel)}  inner-val {_f(sel.get('val_score'), 5)}"
    elif et == "fold_begin":
        wc = ev.get("window_counts", {})
        sp = ev.get("speakers", {})
        s = (f"{ev.get('dataset')} fold {ev.get('fold')}  "
             f"train {wc.get('train', '?')} / val {wc.get('val', '?')} / "
             f"test {wc.get('test', '?')} windows  (test group: {sp.get('test', '?')})")
    elif et == "fold_end":
        s = f"{ev.get('dataset')} fold {ev.get('fold')} finished"
    elif et == "session_begin":
        s = (f"seed {ev.get('seed')}  backend {ev.get('backend')}  "
             f"git {ev.get('git_commit')}  grid≈{ev.get('total_outer_runs')}/fold")
    elif et == "session_end":
        s = f"process done, {ev.get('n_rows', '?')} result rows"
    elif et == "session_error":
        s = f"FAILED: {ev.get('what', '')}"
    else:
        s = json.dumps({k: v for k, v in ev.items()
                        if k not in ("v", "ts_unix", "type", "run_tag", "dataset", "fold")})[:120]
    return t, et, s


# --------------------------------------------------------------------------------------
# plain-text renderer
# --------------------------------------------------------------------------------------
def render_plain(state: SessionState) -> str:
    ln: list[str] = []
    sess = state.session
    c = state.counts()
    elapsed = time.time() - state.started_wall if state.started_wall else None
    total = max(1, c["total"])
    frac = c["done"] / total

    if not sess:
        ln.append(_c("SESSION", "bold", "blue") +
                  "  waiting for the first event... start the run with:")
        ln.append(_c("  EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/"
                     "01_meeting01_run_loso.sh", "dim"))
        return "\n".join(ln)

    done_txt = _c(f"{c['done']} done", "bold", "green")
    running_txt = _c(f"{c['running']} running", "cyan")
    failed_txt = _c(f"{c['failed']} failed", "bold", "red") if c["failed"] else _c("0 failed", "dim")
    ln.append(_c("SESSION", "bold", "blue") +
              f"  {_c(str(sess.get('run_tag', '?')), 'bold')}  seed {sess.get('seed', '?')}  "
              f"backend {sess.get('backend', '?')}  git {sess.get('git_commit', '?')}")
    total_kind = _c("fixed", "green") if state.dataset_roster else _c("estimated", "yellow")
    ln.append(f"  [{_c(_bar(frac, 26), 'cyan')}]  {done_txt}   {running_txt}   {failed_txt}   "
              f"of ~{c['total']} ({total_kind})   elapsed {_hms(elapsed)}   "
              f"eta {_hms(state.eta_seconds())} (rough)")
    for proc in state.procs.values():
        if proc.error:
            ln.append(_c(f"  FAILED  {proc.dataset} fold{proc.fold}: {proc.error}", "bold", "red"))
    if state.loop_error:
        ln.append(_c(f"  poll error (retrying): {state.loop_error}", "yellow"))

    ln.append("")
    ln.append(_c("TRAINING NOW", "bold", "cyan"))
    act = state.active_configs()
    if not act:
        live = state.live_procs()
        if live:
            where = ", ".join(f"{p.dataset} fold {p.fold}" for p in live)
            ln.append(_c(f"  process alive ({where}) — starting the next config "
                        "(no epoch reported yet)", "dim"))
        else:
            ln.append(_c("  nothing training right now (between folds / aggregating / "
                        "not started)", "dim"))
    for a in act[:6]:
        done_ep = len(a.epochs)
        run_ep = a.cur_progress_epoch or (done_ep + 1)
        ep_frac = done_ep / a.max_epochs if a.max_epochs else 0.0
        where_txt = _c(f"{a.dataset} fold {a.fold}", "bold", "cyan")
        ln.append(f"  {where_txt}  {a.model} {a.encoding} "
                  f"{_hp_inline(a.hyperparams)}".rstrip())
        ln.append(f"    epoch {run_ep}/{a.max_epochs} [{_c(_bar(ep_frac), 'cyan')}] {done_ep} done")
        if a.cur_total_batches > 1:
            eta_e = _hms(a.cur_epoch_eta_s) if (a.cur_epoch_eta_s or 0) > 0 else "?"
            bl = f"   loss {_f(a.cur_batch_loss, 5)}" if a.cur_batch_loss is not None else ""
            ln.append(f"    batch {a.cur_batch}/{a.cur_total_batches} "
                      f"[{_c(_bar(a.cur_batch_frac), 'green')}] {a.cur_batch_frac * 100:.0f}%{bl}"
                      f"   ~{eta_e} left this epoch")
        no_improve_style = ("bold", "red") if a.no_improve >= 6 else \
            ("yellow",) if a.no_improve >= 3 else ("dim",)
        best_val_txt = _c(_f(a.running_best_val), "bold", "green")
        no_improve_txt = _c(f"no improvement {a.no_improve} ep", *no_improve_style)
        ln.append(f"    train {_f(a.last_train)}   val {_f(a.last_val)}   "
                  f"best val {best_val_txt} @ ep {a.running_best_epoch}   "
                  f"gap(val-train) {_f(a.gap, 5)}   {no_improve_txt}")
        ln.append(f"    train {_c(_spark([e[1] for e in a.epochs]), 'white')}")
        ln.append(f"    val   {_c(_spark([e[2] for e in a.epochs]), 'white')}")

    ln.append("")
    ln.append(_c("COMPLETED", "bold", "blue") +
              "  (ranked by held-out test loss, else best inner-validation loss)")
    if state.completed_fold_trainings:
        parts = ", ".join(f"{ds} fold {f} ({n} trainings)"
                          for (ds, f), n in sorted(state.completed_fold_trainings.items()))
        ln.append(_c(f"  already complete from an earlier run (RESUME=1 skipped these): {parts}",
                    "yellow"))
    comp = state.completed_configs()
    if not comp:
        if not state.completed_fold_trainings:
            ln.append(_c("  nothing finished yet - the first config takes ~10-20 min after "
                        "a fold starts", "dim"))
    else:
        header = (f"  {'#':>2}  {'model':<14} {'enc':<8} {'hp':<18} {'epochs':>6} "
                  f"{'loss':>10} {'mae':>10} {'train s':>8} {'params':>9}")
        ln.append(_c(header, "dim"))
        best_loss = comp[0].rank_val
        for i, cs in enumerate(comp[:14], 1):
            m = cs.metrics.get("test") or cs.metrics.get("val") or {}
            tms = m.get("train_ms")
            row = (f"  {i:>2}  {cs.model:<14} {cs.encoding:<8} {_hp_str(cs.hyperparams):<18} "
                  f"{cs.epochs_run or len(cs.epochs):>6} {_f(cs.rank_val, 6):>10} "
                  f"{_f(m.get('mae'), 6):>10} "
                  f"{(f'{tms / 1000:.0f}' if tms else '-'):>8} {str(cs.param_count or '-'):>9}")
            ln.append(_c(row, "bold", "green") if cs.rank_val == best_loss else row)

    agg = [r for r in state.aggregation() if r[4] >= 2]
    if agg:
        ln.append("")
        ln.append(_c("MEAN +/- STD", "bold", "blue") + "  of loss over completed seeds x folds")
        for model, enc, mean, std, n_ok, n_fail in agg:
            fail_txt = _c(f", {n_fail} failed", "red") if n_fail else ""
            ln.append(f"  {model:<14} {enc:<8}  {_f(mean, 6)} +/- {_f(std, 6)}  "
                      f"(n={n_ok}{fail_txt})")

    ln.append("")
    ln.append(_c("RECENT", "bold", "blue"))
    for ev in list(state.events)[-8:]:
        t, et, s = _event_line(ev)
        et_txt = _c(f"{et:<15}", *_EVENT_COLOR.get(et, ("white",)))
        ln.append(f"  {_c(t, 'dim')}  {et_txt} {s}")

    ln.append("")
    for legend_line in _legend():
        ln.append(_c(legend_line, "dim"))
    return "\n".join(ln)


# --------------------------------------------------------------------------------------
# rich renderers
# --------------------------------------------------------------------------------------
@dataclass
class DashboardUI:
    """Live-dashboard-only state a hotkey can change: never touched by
    SessionState/render_plain, so --plain and --once stay pure functions of the
    event stream. paused freezes polling (rendering + hotkeys keep working);
    model_filter/fold_focus narrow COMPLETED / TRAINING NOW / the fold grid."""
    paused: bool = False
    model_filter: Optional[str] = None
    fold_focus: Optional[tuple[str, int]] = None

    def cycle_model_filter(self, state: SessionState, step: int = 1) -> None:
        """Advance model_filter by `step` through [None, *sorted distinct models
        seen], wrapping around -- None means "no filter"."""
        models = sorted({c.model for c in state.configs.values()})
        choices: list[Optional[str]] = [None] + models
        i = choices.index(self.model_filter) if self.model_filter in choices else 0
        self.model_filter = choices[(i + step) % len(choices)]

    def cycle_fold_focus(self, state: SessionState, step: int = 1) -> None:
        """Advance fold_focus by `step` through [None, *sorted (dataset, fold)
        pairs in state.fold_grid()], wrapping around -- None means "no focus"."""
        folds = sorted(state.fold_grid().keys())
        choices: list[Optional[tuple[str, int]]] = [None] + folds
        i = choices.index(self.fold_focus) if self.fold_focus in choices else 0
        self.fold_focus = choices[(i + step) % len(choices)]

    def clear(self) -> None:
        """Drop both filters (the 'c' hotkey) -- paused is untouched."""
        self.model_filter = None
        self.fold_focus = None


class _KeyReader:
    """Non-blocking single-key reads from a TTY stdin for the live dashboard's
    hotkeys (p pause, f/F cycle model filter, [/] cycle fold focus, c clear,
    q quit). Uses cbreak mode (stdlib termios/tty), not raw, so Ctrl-C still
    raises KeyboardInterrupt normally. A complete no-op (poll() always returns
    "") when stdin is not a terminal (CI, a pipe, `--once`'s non-dashboard path)
    -- the dashboard must never require a keyboard to run."""

    def __init__(self) -> None:
        """Detect once whether stdin is a TTY; every other method is a no-op
        when it is not, so this never needs to raise."""
        self._enabled = sys.stdin.isatty()
        self._fd: Optional[int] = None
        self._old: Optional[list[Any]] = None

    def __enter__(self) -> "_KeyReader":
        """Switch stdin to cbreak mode (no line buffering, no local echo)."""
        if self._enabled:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc: Any) -> None:
        """Restore stdin's original terminal settings, even on an exception."""
        if self._enabled and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> str:
        """One buffered keystroke if one is waiting, else "" -- never blocks."""
        if not self._enabled:
            return ""
        import select
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if ready else ""
def _panel_session(state: SessionState, ui: Optional[DashboardUI] = None):  # noqa: ANN201
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    sess = state.session
    c = state.counts()
    elapsed = time.time() - state.started_wall if state.started_wall else None
    total = max(1, c["total"])
    frac = c["done"] / total

    g = Table.grid(padding=(0, 1))
    g.add_column(justify="left")
    if not sess:
        g.add_row(Text("waiting for the first event... start the run with:", style="bold yellow"))
        g.add_row(Text("  EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/"
                       "01_meeting01_run_loso.sh", style="dim"))
    else:
        header = Text.assemble(
            (" " + str(sess.get("run_tag", "?")) + " ", "bold black on bright_white"),
            (f"   seed {sess.get('seed', '?')}   backend {sess.get('backend', '?')}"
             f"   git {sess.get('git_commit', '?')}", "dim"),
        )
        if ui and ui.paused:
            header.append_text(Text("  PAUSED ", style="bold black on yellow"))
        g.add_row(header)
        # badges use a solid background so done/running/failed pop at a glance,
        # not just a color change on plain text
        failed_badge = (f" {c['failed']} FAILED ", "bold white on red") if c["failed"] \
            else (" 0 failed ", "dim")
        g.add_row(Text.assemble(
            (f"[{_bar(frac, 26)}] ", "green"),
            (f" {c['done']} done ", "bold black on green"),
            ("  ", ""),
            (f" {c['running']} running ", "bold black on cyan"),
            ("  ", ""),
            failed_badge,
            (f"   of ~{c['total']} trainings ({frac * 100:.1f}%) ", "dim"),
            (" fixed " if state.dataset_roster else " estimated ",
             "bold black on green" if state.dataset_roster else "bold black on yellow"),
        ))
        g.add_row(Text(f"elapsed {_hms(elapsed)}   eta {_hms(state.eta_seconds())} (rough, "
                       f"from completed-so-far rate)", style="dim"))
        sp = sess.get("search_space", {})
        n_e = len(sp.get("encodings", []) or [])
        n_b = len(sp.get("baselines", []) or [])
        has_snn = bool(sp.get("snn_architectures"))
        ga_desc = (f"GA pop={sp.get('ga_population_size', '?')}×"
                   f"(1+{sp.get('ga_generations', '?')} gen)" if has_snn else "no SNN arm")
        g.add_row(Text(
            f"grid: {n_b} baselines × {n_e} encodings + {ga_desc} + retrain, "
            f"× {sess.get('repeats', '?')} seeds  "
            f"≈ {state.per_fold_trainings()}/fold", style="dim"))
        fails = [p for p in state.procs.values() if p.error]
        if fails:
            g.add_row(Text("  ".join(f"[FAILED {p.dataset} f{p.fold}] {p.error}" for p in fails),
                           style="bold white on red"))
        g.add_row(Text.assemble(
            ("keys: ", "bold dim"),
            ("p", "bold"), (" pause/resume   ", "dim"),
            ("f", "bold"), ("/", "dim"), ("F", "bold"), (" cycle model filter   ", "dim"),
            ("[", "bold"), ("/", "dim"), ("]", "bold"), (" cycle fold focus   ", "dim"),
            ("c", "bold"), (" clear filters   ", "dim"),
            ("q", "bold"), (" quit", "dim"),
        ))
    return Panel(g, title="[bold]SESSION[/bold]", title_align="left", border_style="bold blue",
                 padding=(0, 1))


def _panel_now(state: SessionState, ui: Optional[DashboardUI] = None):  # noqa: ANN201
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    def _trend_color(values: list[Optional[float]]) -> str:
        """green if the loss fell since the first finite point, red if it rose,
        white if flat or too few points to tell -- a loss curve reads at a glance."""
        vals = [v for v in values if v is not None and math.isfinite(v)]
        if len(vals) < 2:
            return "white"
        return "green" if vals[-1] < vals[0] else "red" if vals[-1] > vals[0] else "white"

    act = state.active_configs()
    if ui and ui.fold_focus:
        act = [a for a in act if (a.dataset, a.fold) == ui.fold_focus]
    if not act:
        live = state.live_procs()
        if ui and ui.fold_focus:
            body = Text(f"Nothing training in fold focus {ui.fold_focus[0]} fold "
                        f"{ui.fold_focus[1]} right now.\nPress [ or ] to change focus, "
                        "c to clear it.", style="bold yellow")
        elif live:
            where = ", ".join(f"{p.dataset} fold {p.fold}" for p in live)
            body = Text(f"Process alive ({where}) — starting the next config.\n"
                        "The fast SNN sweep briefly shows this between configs; "
                        "an epoch will appear within a few seconds.", style="bold cyan")
        else:
            body = Text("Nothing is training right now.\n"
                        "The run may be between folds, running the Python aggregation, "
                        "or not started yet.", style="bold yellow")
        return Panel(body, title="[bold]TRAINING NOW[/bold]", title_align="left",
                     border_style="grey50", padding=(0, 1))

    blocks = []
    for idx, a in enumerate(act[:3]):
        done_ep = len(a.epochs)
        run_ep = a.cur_progress_epoch or (done_ep + 1)
        ep_frac = done_ep / a.max_epochs if a.max_epochs else 0.0
        avg = a.avg_epoch_ms()
        eta_tr = (_hms((a.max_epochs - done_ep) * avg / 1000.0)
                  if avg and a.max_epochs else None)
        bar_color = "green" if ep_frac >= 0.9 else "cyan"
        t = Table.grid(padding=(0, 1))
        t.add_column()
        hp = _hp_inline(a.hyperparams)
        t.add_row(Text.assemble(
            (f" {a.dataset} fold {a.fold} ", "bold black on cyan"),
            (f"   {a.model}  {a.encoding}{('  ' + hp) if hp else ''}", "bold"),
        ))
        t.add_row(Text.assemble(
            (f"epoch {run_ep}/{a.max_epochs}  ", ""),
            (f"[{_bar(ep_frac, 22)}] ", bar_color),
            (f"{done_ep} done", "dim"),
            (f"    ~{eta_tr} left in this training" if eta_tr else "", "dim"),
        ))
        if a.cur_total_batches > 1:
            eta_e = _hms(a.cur_epoch_eta_s) if (a.cur_epoch_eta_s or 0) > 0 else "?"
            bl = (f"  loss {_f(a.cur_batch_loss, 5)}"
                  if a.cur_batch_loss is not None else "")
            t.add_row(Text.assemble(
                (f"  batch {a.cur_batch}/{a.cur_total_batches} ", ""),
                (f"[{_bar(a.cur_batch_frac, 22)}] ", "green"),
                (f"{a.cur_batch_frac * 100:.0f}%{bl}   ~{eta_e} left this epoch", "dim"),
            ))
        t.add_row(Text.assemble(
            (f"train {_f(a.last_train)}    val {_f(a.last_val)}    "),
            (f"best val {_f(a.running_best_val)} @ ep {a.running_best_epoch}", "bold green"),
        ))
        # a rising count of epochs since the best val loss is a fact worth a glance,
        # not a diagnosis -- the color only escalates how loudly it is shown.
        no_improve_style = "bold white on red" if a.no_improve >= 6 else \
            "bold yellow" if a.no_improve >= 3 else "dim"
        t.add_row(Text.assemble(
            (f"gap (val - train) {_f(a.gap, 5)}    ", "dim"),
            (f" no improvement for {a.no_improve} epoch(s) ", no_improve_style),
            (f"    val loss rose {a.val_increased_epochs} of the last epochs", "dim"),
        ))
        train_vals = [e[1] for e in a.epochs]
        val_vals = [e[2] for e in a.epochs]
        if idx == 0:
            # the primary (most-recently-active) training gets the full-height
            # line chart (reuses _ascii_plot, the same renderer --rank N uses --
            # one chart implementation, not a second one); the rest fall back to
            # a one-line sparkline so 2-3 concurrent folds don't blow the layout.
            for label, vals in (("train", train_vals), ("val", val_vals)):
                color = _trend_color(vals)
                t.add_row(Text(f"{label} loss:", style="dim"))
                for row in _ascii_plot(vals, width=44, height=4):
                    t.add_row(Text(row, style=color))
        else:
            hint = "" if len(a.epochs) >= 3 else "   (fills in as epochs complete)"
            t.add_row(Text.assemble(("train  ", "dim"),
                                    (_spark(train_vals) or "·", _trend_color(train_vals)),
                                    (hint, "dim")))
            t.add_row(Text.assemble(("val    ", "dim"),
                                    (_spark(val_vals) or "·", _trend_color(val_vals))))
        blocks.append(t)
    return Panel(Group(*blocks), title="[bold]TRAINING NOW[/bold]", title_align="left",
                 border_style="bold cyan", padding=(0, 1))


def _panel_ranking(state: SessionState, max_rows: int = 12,
                   ui: Optional[DashboardUI] = None):  # noqa: ANN201
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    comp = state.completed_configs()
    filter_bits = []
    if ui and ui.model_filter:
        comp = [c for c in comp if c.model == ui.model_filter]
        filter_bits.append(f"model={ui.model_filter}")
    if ui and ui.fold_focus:
        comp = [c for c in comp if (c.dataset, c.fold) == ui.fold_focus]
        filter_bits.append(f"fold={ui.fold_focus[0]} f{ui.fold_focus[1]}")
    filter_note = f"  [yellow](filtered: {', '.join(filter_bits)} -- press c to clear)[/yellow]" \
        if filter_bits else ""
    title = ("[bold]COMPLETED[/bold]  —  ranked by held-out test loss "
            f"(else best inner-validation loss){filter_note}")
    resumed_note = ""
    if state.completed_fold_trainings:
        parts = ", ".join(f"{ds} fold {f} ({n} trainings)"
                          for (ds, f), n in sorted(state.completed_fold_trainings.items()))
        resumed_note = f"Already complete from an earlier run (RESUME=1 skipped these): {parts}.\n"
    if not comp:
        msg = "No config matches the current filter.\n" if filter_bits else \
            (resumed_note +
             ("No other config has finished yet.\n" if resumed_note else "No config has finished yet.\n") +
             "Per fold: 3 baselines + a 27-config SNN v_th×α×arch sweep + "
             "1 retrain,  × 3 encodings × 5 seeds.\n"
             "The first (LSTM-AE) result lands ~10–20 min after a fold starts.")
        return Panel(Text(msg, style="bold yellow"), title=title, title_align="left",
                     border_style="grey50", padding=(0, 1))

    tbl = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, header_style="bold magenta")
    for name, just in (("#", "right"), ("model", "left"), ("enc", "left"), ("hp", "left"),
                       ("ep", "right"), ("loss", "right"), ("kind", "left"),
                       ("mae", "right"), ("train s", "right"), ("params", "right")):
        tbl.add_column(name, justify=just, no_wrap=True)
    best_loss = comp[0].rank_val
    # kind = which split produced this row's loss: "test" (held-out, most trustworthy),
    # "val" (inner-validation, evaluated but not held-out), "best" (no evaluation yet,
    # just the lowest training-time validation loss seen).
    kind_style = {"test": "bold green", "val": "yellow", "best": "dim"}
    for i, cs in enumerate(comp[:max_rows], 1):
        m = cs.metrics.get("test") or cs.metrics.get("val") or {}
        tms = m.get("train_ms")
        is_best = cs.rank_val == best_loss
        # zebra striping (even rows on a faint background) makes a 10+ row table
        # scannable at a glance; the best row overrides it with a solid highlight.
        if is_best:
            style = "bold black on green"
        elif i % 2 == 0:
            style = "on grey15"
        else:
            style = ""
        rank_cell = f"★{i}" if is_best else str(i)
        tbl.add_row(
            rank_cell, cs.model, cs.encoding, _hp_str(cs.hyperparams),
            str(cs.epochs_run or len(cs.epochs)),
            _f(cs.rank_val, 6),
            cs.rank_val_kind if is_best else Text(cs.rank_val_kind, style=kind_style.get(cs.rank_val_kind, "")),
            _f(m.get("mae"), 6),
            f"{tms / 1000:.0f}" if tms else "-",
            str(cs.param_count or "-"),
            style=style,
        )
    extra = "" if len(comp) <= max_rows else f"   (+{len(comp) - max_rows} more — --rank N for detail)"
    parts = ([Text(resumed_note.strip(), style="bold yellow")] if resumed_note else []) + \
        [tbl, Text.assemble(
            ("★", "bold green"), (" = best so far.   ", "dim"),
            (f"monitor.py --rank N  for one row's full detail{extra}   ", "dim"),
            ("kind", "bold"), (": ", "dim"), ("test", "bold green"), ("=held-out, ", "dim"),
            ("val", "yellow"), ("=evaluated but not held-out, ", "dim"),
            ("best", "dim"), ("=training-time only", "dim"))]

    marg = state.marginals()
    if any(any(n for _, _, n in rs) for rs in marg.values()):
        mt = Table.grid(padding=(0, 2))
        mt.add_column(style="bold dim")
        mt.add_column()
        for dim, rlist in marg.items():
            cells = "   ".join(f"{label} {_f(best, 5)} (n={n})" for label, best, n in rlist if n)
            if cells:
                mt.add_row(dim, cells)
        parts += [Text(""),
                  Text("best inner-val loss per SNN dimension "
                       "(descriptive summary — not a causal claim)", style="italic dim"),
                  mt]

    agg = [r for r in state.aggregation() if r[4] >= 2]
    if agg:
        parts.append(Text(""))
        parts.append(Text("loss  mean ± std  over completed seeds × folds:", style="italic dim"))
        for model, enc, mean, std, n_ok, n_fail in agg:
            parts.append(Text(f"  {model} {enc}: {_f(mean, 5)} ± {_f(std, 5)} "
                              f"(n={n_ok}{f', {n_fail} failed' if n_fail else ''})", style="dim"))

    return Panel(Group(*parts), title=title, title_align="left", border_style="bold magenta",
                 padding=(0, 1))


def _panel_fold_grid(state: SessionState, dash_ui: Optional[DashboardUI] = None):  # noqa: ANN201
    """One row per dataset, one cell per fold: the whole nested-LOSO sweep's
    state at a glance -- which folds are done, mid-run, stuck, or not started
    yet -- instead of only whichever fold TRAINING NOW happens to show. Press
    [ / ] to focus one cell (narrows TRAINING NOW + COMPLETED to it), c clears."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    grid = state.fold_grid()
    n_fold = int(state.session.get("cv_num_folds", 1) or 1)
    datasets = sorted({dataset for dataset, _ in grid} |
                      set(state.session.get("all_datasets", []) or []))
    if not datasets or n_fold <= 1:
        body = Text("Single fold (or no session yet) -- nothing to overview.", style="dim")
        return Panel(body, title="[bold]FOLD GRID[/bold]", title_align="left",
                     border_style="grey50", padding=(0, 1))

    cell_style = {"done": "bold black on green", "running": "bold black on cyan",
                 "failed": "bold white on red", "unstarted": "dim"}
    cell_char = {"done": "done", "running": "run ", "failed": "FAIL", "unstarted": " -- "}
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="bold", no_wrap=True)
    for _ in range(n_fold):
        tbl.add_column(justify="center", no_wrap=True)
    for dataset in datasets:
        row: list[Any] = [dataset]
        for fold in range(n_fold):
            cell = grid.get((dataset, fold))
            status = cell.status if cell else "unstarted"
            pct = f"{cell.frac * 100:3.0f}%" if cell and cell.total else "  - "
            style = cell_style[status]
            if dash_ui and dash_ui.fold_focus == (dataset, fold):
                style += " underline"
            row.append(Text(f" f{fold} {cell_char[status]} {pct} ", style=style))
        tbl.add_row(*row)
    return Panel(tbl, title="[bold]FOLD GRID[/bold]  —  every (dataset, fold) in the sweep",
                 title_align="left", border_style="bold yellow", padding=(0, 1))
def _panel_events(state: SessionState, n: int = 8):  # noqa: ANN201
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    colour = {"session_error": "bold red", "config_end": "green", "config_selected": "yellow",
              "fold_begin": "cyan", "fold_end": "cyan", "train_end": "bold green",
              "session_begin": "magenta"}
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_column(overflow="ellipsis")
    for ev in list(state.events)[-n:]:
        tt, et, s = _event_line(ev)
        t.add_row(tt, Text(et, style=colour.get(et, "white")), s)
    if not state.events:
        t.add_row("", "", Text("(no events yet)", style="dim"))
    return Panel(t, title="[bold]RECENT[/bold]", title_align="left", border_style="bold green",
                 padding=(0, 1))


def _panel_legend():  # noqa: ANN201
    """A standalone, always-visible key so no abbreviation on the dashboard ever
    has to be guessed at -- kept as its own panel (not squeezed into SESSION)
    so it stays legible instead of wrapping into a wall of dim text."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    t = Table.grid(padding=(0, 1))
    t.add_column(style="bold", no_wrap=True)
    t.add_column()
    rows = [
        ("val / train", "loss for the current epoch (validation / training split)"),
        ("gap", "val − train, last epoch -- a spread, not a diagnosis"),
        ("no improve", "epochs since best val (dim → yellow ≥3 → red ≥6)"),
        ("best val @ep", "lowest val loss seen, and which epoch produced it"),
        ("loss (table)", "held-out TEST once evaluated, else best inner-val"),
        ("kind", Text.assemble(("test", "bold green"), ("=held-out  ", ""),
                              ("val", "yellow"), ("=evaluated, not held-out  ", ""),
                              ("best", "dim"), ("=training-time only", ""))),
        ("★", "the best-ranked row so far"),
        ("mae", "mean absolute error, same split as loss"),
        ("ETA", "trainings-left / (done-so-far rate) -- rough, not a guarantee"),
        ("~N", "approximate count (grid size can shift under RESUME)"),
    ]
    for label, desc in rows:
        t.add_row(label, desc)
    return Panel(t, title="[bold]LEGEND[/bold]", title_align="left", border_style="grey50",
                 padding=(0, 1))


def render_dashboard(state: SessionState, height: int = 40,
                     dash_ui: Optional[DashboardUI] = None):  # noqa: ANN201
    """Full-terminal layout: SESSION (identity + counters + hotkeys) and
    TRAINING NOW (the active fold's full loss chart) at fixed heights, a FOLD
    GRID row overviews the whole nested-LOSO sweep, COMPLETED takes the slack,
    and a bottom row splits RECENT (left) and a standing LEGEND (right). Each
    region clips its content, so a short terminal just shows fewer ranking /
    event rows — nothing overflows."""
    from rich.layout import Layout
    from rich.panel import Panel

    dash_ui = dash_ui or DashboardUI()

    def _safe(fn, *a):  # noqa: ANN001
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001
            return Panel(f"[red]panel error:[/red] {exc}", border_style="red")

    ev_rows = max(3, min(9, height - 34))
    grid = state.fold_grid()
    datasets = {ds for ds, _ in grid} | set(state.session.get("all_datasets", []) or [])
    n_fold = int(state.session.get("cv_num_folds", 1) or 1)
    grid_size = (max(1, len(datasets)) + 2) if n_fold > 1 else 3
    rank_rows = max(3, height - 8 - 19 - grid_size - (ev_rows + 2) - 2)

    root = Layout()
    root.split_column(
        Layout(_safe(_panel_session, state, dash_ui), name="session", size=8),
        Layout(_safe(_panel_now, state, dash_ui), name="now", size=19),
        Layout(_safe(_panel_fold_grid, state, dash_ui), name="grid", size=grid_size),
        Layout(_safe(_panel_ranking, state, rank_rows, dash_ui), name="done", ratio=1, minimum_size=5),
        Layout(name="bottom", size=ev_rows + 2),
    )
    root["bottom"].split_row(
        Layout(_safe(_panel_events, state, ev_rows), name="recent", ratio=2),
        Layout(_safe(_panel_legend), name="legend", ratio=1, minimum_size=26),
    )
    return root


def render_detail(cfg: ConfigState, session: dict[str, Any]):  # noqa: ANN201
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    head = Table.grid(padding=(0, 1))
    head.add_column(style="dim")
    head.add_column()
    head.add_row("config", cfg.config_id)
    head.add_row("model / encoding / role", f"{cfg.model} / {cfg.encoding} / {cfg.role}")
    head.add_row("hyperparameters", json.dumps(cfg.hyperparams) or "-")
    head.add_row("status", cfg.status)

    repro = Table.grid(padding=(0, 1))
    repro.add_column(style="dim")
    repro.add_column()
    repro.add_row("seed / fold / run", f"{cfg.seed} / {cfg.fold} / {cfg.run_id}")
    repro.add_row("dataset", cfg.dataset)
    repro.add_row("config_hash", str(session.get("config_hash", "-")))
    repro.add_row("git_commit", str(session.get("git_commit", "-")))
    repro.add_row("backend", str(session.get("backend", "-")))
    repro.add_row("learning rate", _f(cfg.lr, 6) if cfg.lr else "-")

    train = Table.grid(padding=(0, 1))
    train.add_column(style="dim")
    train.add_column()
    train.add_row("epochs run / max", f"{cfg.epochs_run or len(cfg.epochs)} / {cfg.max_epochs}")
    train.add_row("stop reason", cfg.stop_reason or "-")
    train.add_row("best inner-val loss", f"{_f(cfg.running_best_val)} @ epoch {cfg.running_best_epoch}")
    train.add_row("final train / val", f"{_f(cfg.last_train)} / {_f(cfg.last_val)}")
    train.add_row("train/val gap (last epoch)", _f(cfg.gap, 6))
    train.add_row("val loss rose (recent epochs)", str(cfg.val_increased_epochs))
    train.add_row("no improvement for", f"{cfg.no_improve} epoch(s)")
    avg = cfg.avg_epoch_ms()
    train.add_row("avg epoch", f"{avg / 1000:.2f} s" if avg else "-")

    metric_tbls = []
    for split, mm in cfg.metrics.items():
        mt = Table.grid(padding=(0, 1))
        mt.add_column(style="dim")
        mt.add_column()
        for k, v in mm.items():
            mt.add_row(k, str(v))
        metric_tbls.append(Panel(mt, title=f"metrics [{split}]", title_align="left",
                                 border_style="green", padding=(0, 1)))

    plots = [Text("train loss", style="dim")]
    plots += [Text(r) for r in _ascii_plot([e[1] for e in cfg.epochs])]
    plots += [Text(""), Text("val loss", style="dim")]
    plots += [Text(r) for r in _ascii_plot([e[2] for e in cfg.epochs])]

    return Group(
        Panel(head, title="CONFIG", title_align="left", border_style="cyan", padding=(0, 1)),
        Panel(repro, title="REPRODUCIBILITY", title_align="left", border_style="blue",
              padding=(0, 1)),
        Panel(train, title="TRAINING", title_align="left", border_style="blue", padding=(0, 1)),
        *metric_tbls,
        Panel(Group(*plots), title="LOSS CURVES", title_align="left", border_style="blue",
              padding=(0, 1)),
    )


def _drain(results_dir: str, run_tag: str, profile_path: Optional[str] = None) -> SessionState:
    tailer = EventTailer(results_dir, run_tag)
    state = SessionState()
    state.set_dataset_roster(dataset_roster_from_profile(profile_path or default_profile_path(run_tag)))
    for ev in tailer.poll():
        state.apply(ev)
    state.reconcile(tailer.mtimes, tailer.missing)
    state.note_completed_folds(scan_completed_folds(results_dir, run_tag))
    return state


def run_detail(results_dir: str, run_tag: str, rank: int) -> int:
    try:
        from rich.console import Console
    except ModuleNotFoundError:
        print("rich is not installed (see scripts/requirements.txt).", file=sys.stderr)
        return 2
    state = _drain(results_dir, run_tag)
    comp = state.completed_configs()
    if not comp:
        print("No completed configs yet.")
        return 1
    if rank < 1 or rank > len(comp):
        print(f"--rank must be 1..{len(comp)} (that many configs have finished).")
        return 1
    Console().print(render_detail(comp[rank - 1], state.session))
    return 0


def _handle_hotkey(key: str, dash_ui: DashboardUI, state: SessionState) -> bool:
    """Apply one keypress from _KeyReader.poll() to dash_ui. Returns True if
    the dashboard should quit (the 'q' hotkey) -- everything else is a no-op
    for an unrecognized or empty (no key waiting) `key`."""
    if key in ("q", "Q"):
        return True
    if key in ("p", "P"):
        dash_ui.paused = not dash_ui.paused
    elif key == "f":
        dash_ui.cycle_model_filter(state, 1)
    elif key == "F":
        dash_ui.cycle_model_filter(state, -1)
    elif key == "[":
        dash_ui.cycle_fold_focus(state, -1)
    elif key == "]":
        dash_ui.cycle_fold_focus(state, 1)
    elif key in ("c", "C"):
        dash_ui.clear()
    return False


def _poll_once(tailer: EventTailer, state: SessionState, results_dir: str, run_tag: str,
               consecutive_errors: int) -> int:
    """One tailer.poll() -> state.apply() round, tolerating a transient FS
    race (a results file swapped/removed by a concurrent git op, a half-
    written line) so it never freezes the dashboard. Raises once the same
    error has recurred on 30 consecutive polls instead of forever. Returns
    the (possibly reset) consecutive-error count."""
    try:
        for event in tailer.poll():
            state.apply(event)
        state.reconcile(tailer.mtimes, tailer.missing)
        state.note_completed_folds(scan_completed_folds(results_dir, run_tag))
        return 0
    except Exception as exc:  # noqa: BLE001
        consecutive_errors += 1
        state.loop_error = f"{type(exc).__name__}: {exc}"
        if consecutive_errors >= 30:
            raise
        return consecutive_errors


def run_dashboard(results_dir: str, run_tag: str, poll: float,
                  profile_path: Optional[str] = None) -> int:
    try:
        from rich.console import Console
        from rich.live import Live
    except ModuleNotFoundError:
        print("rich is not installed. It is in scripts/requirements.txt — rerun "
              "`cmake --preset=max-performance`, or use --plain.", file=sys.stderr)
        return 2

    tailer = EventTailer(results_dir, run_tag)
    state = SessionState()
    state.set_dataset_roster(dataset_roster_from_profile(profile_path or default_profile_path(run_tag)))
    dash_ui = DashboardUI()
    console = Console()
    try:
        with Live(render_dashboard(state, console.size.height, dash_ui), console=console,
                  screen=True, refresh_per_second=4, redirect_stderr=False) as live, \
                _KeyReader() as keys:
            consecutive_errors = 0
            while True:
                ch = keys.poll()
                if _handle_hotkey(ch, dash_ui, state):
                    return 0
                if not dash_ui.paused:
                    consecutive_errors = _poll_once(tailer, state, results_dir, run_tag,
                                                    consecutive_errors)
                live.update(render_dashboard(state, console.size.height, dash_ui))
                time.sleep(max(0.5, poll) if not ch else 0.05)
    except KeyboardInterrupt:
        return 0


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------
def _self_test() -> int:
    _set_color(False)  # deterministic text for the substring checks below

    def ev(**kw: Any) -> dict[str, Any]:
        kw.setdefault("v", 1)
        kw.setdefault("ts_unix", time.time())
        kw.setdefault("run_tag", "t")
        return kw

    st = SessionState()
    st.apply(ev(type="session_begin", dataset="fsdd", fold=0, seed=42, repeats=2,
               cv_num_folds=6, all_datasets=["fsdd"],
               search_space={"snn_architectures": ["dense"], "ga_population_size": 1,
                             "ga_generations": 0, "encodings": ["direct"],
                             "baselines": ["lstm-ae"]}))
    st.apply(ev(type="fold_begin", dataset="fsdd", fold=0))
    st.apply(ev(type="config_begin", dataset="fsdd", fold=0, config_id="lstm-ae_direct_seed42_run1",
               model="lstm-ae", encoding="direct", role="baseline", run_id=1, seed=42,
               max_epochs=10))
    for e, (tr, va) in enumerate([(0.5, 0.6), (0.3, 0.4), (0.2, 0.35), (0.15, 0.37), (0.1, 0.39)], 1):
        st.apply(ev(type="epoch", dataset="fsdd", fold=0,
                   config_id="lstm-ae_direct_seed42_run1", epoch=e, max_epochs=10,
                   train_loss=tr, val_loss=va, epoch_ms=1000.0))
    st.apply(ev(type="train_end", dataset="fsdd", fold=0,
               config_id="lstm-ae_direct_seed42_run1", epochs_run=5, stop_reason="early_stop",
               best_val_loss=0.35, best_val_epoch=3))
    st.apply(ev(type="config_end", dataset="fsdd", fold=0,
               config_id="lstm-ae_direct_seed42_run1", model="lstm-ae", encoding="direct",
               role="baseline", run_id=1, seed=42, split="test",
               metrics={"mse": 0.33, "mae": 0.4, "param_count": 1234, "macs": 5678}))

    c = st.configs["lstm-ae_direct_seed42_run1"]
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    check(c.status == "done", "config marked done after config_end")
    check(abs((c.running_best_val or 0) - 0.35) < 1e-9, "running best val = 0.35")
    check(c.running_best_epoch == 3, "best epoch = 3")
    check(c.no_improve == 2, "no_improve = 2 (epochs 4,5 after best at 3)")
    check(c.val_increased_epochs == 2, "val rose for last 2 epochs")
    check(abs((c.gap or 0) - (0.39 - 0.1)) < 1e-9, "gap = val - train on last epoch")
    check(abs((c.rank_val or 0) - 0.33) < 1e-9, "rank_val uses test mse when present")
    check(c.rank_val_kind == "test", "rank_val_kind reports 'test'")
    check(c.param_count == 1234, "param_count carried from config_end metrics")
    check(st.counts()["done"] == 1 and st.counts()["running"] == 0, "counts: 1 done, 0 running")

    st.apply(ev(type="config_begin", dataset="fsdd", fold=0, config_id="lstm-ae_direct_seed43_run2",
               model="lstm-ae", encoding="direct", role="baseline", run_id=2, seed=43,
               max_epochs=10))
    st.apply(ev(type="config_end", dataset="fsdd", fold=0,
               config_id="lstm-ae_direct_seed43_run2", model="lstm-ae", encoding="direct",
               role="baseline", run_id=2, seed=43, split="test", metrics={"mse": 0.37}))
    agg = {(m, e): (mean, std, n) for m, e, mean, std, n, _ in st.aggregation()}
    mean, std, n = agg[("lstm-ae", "direct")]
    check(n == 2 and abs(mean - 0.35) < 1e-9, "aggregation mean of 0.33 & 0.37 = 0.35")
    check(std is not None and abs(std - math.sqrt(((0.33 - 0.35) ** 2 + (0.37 - 0.35) ** 2))) < 1e-9,
          "aggregation sample std correct")

    st.apply(ev(type="config_begin", dataset="fsdd", fold=1, config_id="gru-ae_direct_seed42_run1",
               model="gru-ae", encoding="direct", role="baseline", run_id=1, seed=42,
               max_epochs=10, _path="/tmp/x_fsdd_fold1_events.jsonl"))
    st.apply(ev(type="session_error", dataset="fsdd", fold=1, what="boom",
               _path="/tmp/x_fsdd_fold1_events.jsonl"))
    st.reconcile({})
    check(st.configs["gru-ae_direct_seed42_run1"].status == "failed",
          "running config of an errored process -> failed")
    check(st.counts()["failed"] == 1, "counts: 1 failed")

    st.apply(ev(type="epoch", dataset="fsdd", fold=0, config_id="lstm-ae_direct_seed42_run1",
               epoch=6, max_epochs=10, train_loss=None, val_loss=None))
    st.apply(ev(type="totally_unknown_event", dataset="fsdd", fold=0))
    st.apply({"type": "epoch", "v": 999})
    check(True, "unknown event type / null metrics / wrong version tolerated (no exception)")

    # within-epoch progress: populated by epoch_progress, cleared by the epoch event
    st2 = SessionState()
    st2.apply(ev(type="config_begin", dataset="fsdd", fold=0, config_id="cX", model="snn-ae",
                encoding="direct", role="snn_sweep", run_id=1, seed=42, max_epochs=30))
    st2.apply(ev(type="epoch_progress", dataset="fsdd", fold=0, config_id="cX", epoch=1,
                max_epochs=30, batch=60, total_batches=200, frac=0.3, batch_loss=0.4,
                epoch_elapsed_s=9.0, epoch_eta_s=21.0))
    cx = st2.configs["cX"]
    check(cx.cur_batch == 60 and cx.cur_total_batches == 200 and cx.cur_progress_epoch == 1,
          "epoch_progress populates within-epoch batch fields")
    check(st2 in (st2,) and cx in st2.active_configs(),
          "a config with only batch progress (slow first epoch) is 'active'")
    check(not any(e.get("type") == "epoch_progress" for e in st2.events),
          "epoch_progress is not written to the event log")
    st2.apply(ev(type="epoch", dataset="fsdd", fold=0, config_id="cX", epoch=1, max_epochs=30,
                train_loss=0.3, val_loss=0.35, epoch_ms=30000.0))
    check(cx.cur_total_batches == 0, "the epoch event clears within-epoch progress")

    txt = render_plain(st)
    banned = ["overfit", "converged", "convergence achieved", "significant", "generalizes",
              "generalisation is", "the model is better"]
    check(all(b not in txt.lower() for b in banned), "plain render has no inferential language")
    check("COMPLETED" in txt and "TRAINING NOW" in txt, "plain render has the core sections")
    for ln in list(st.events):
        _event_line(ln)
    check(True, "_event_line handles every event type without raising")

    # A fold's events file disappearing (RESUME=1 skipping an already-complete
    # fold, or a crashed attempt's leftover cleaned up by hand) must fail its
    # proc/config on the very next poll — not linger as "running" for up to
    # STALE_SECONDS (15 min) the way a merely-slow file would. Uses a real file
    # on disk since EventTailer globs the filesystem, not an in-memory fixture.
    with tempfile.TemporaryDirectory() as td:
        fold_path = os.path.join(td, "t_fsdd_fold9_events.jsonl")
        with open(fold_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(ev(type="config_begin", dataset="fsdd", fold=9,
                                    config_id="lstm-ae_direct_seed42_run1", model="lstm-ae",
                                    encoding="direct", role="baseline", run_id=1, seed=42,
                                    max_epochs=10)) + "\n")
        tailer = EventTailer(td, "t")
        st3 = SessionState()
        for e in tailer.poll():
            st3.apply(e)
        st3.reconcile(tailer.mtimes, tailer.missing)
        check(not tailer.missing, "no missing files on the first poll")
        check(st3.configs["lstm-ae_direct_seed42_run1"].status == "running",
              "config is running while its events file still exists")

        os.remove(fold_path)
        for e in tailer.poll():  # nothing new to read, but this is what notices the removal
            st3.apply(e)
        st3.reconcile(tailer.mtimes, tailer.missing)
        check(fold_path in tailer.missing, "EventTailer reports the removed file as missing")
        check(st3.configs["lstm-ae_direct_seed42_run1"].status == "failed",
              "config fails immediately when its events file is removed, no STALE_SECONDS wait")
        check(not st3.live_procs(), "the removed-file proc no longer counts as live")

    # RESUME=1 skips a fold whose *_comparative_metrics.csv already exists —
    # scan_completed_folds() must credit those trainings towards "done" even
    # though this session never saw a single live event for that fold.
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "t_fsdd_fold0_comparative_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "encoding", "architecture", "v_th", "alpha", "run", "seed", "split"])
            # two rows (val + test) of the SAME training -> must count once
            w.writerow(["lstm-ae", "direct", "lstm", "0", "0", "1", "42", "val"])
            w.writerow(["lstm-ae", "direct", "lstm", "0", "0", "1", "42", "test"])
            w.writerow(["gru-ae", "direct", "gru", "0", "0", "1", "42", "val"])
        found = scan_completed_folds(td, "t")
        check(found == {("fsdd", 0): 2}, "scan_completed_folds counts distinct trainings, not raw rows")

        st4 = SessionState()
        st4.note_completed_folds(found)
        check(st4.counts()["done"] == 2, "resumed fold's trainings count towards done")

        # a fold THIS session watched live must not be double-counted from its CSV
        st4.folds_seen.add(("fsdd", 0))
        st4.note_completed_folds(found)
        check(st4.counts()["done"] == 0,
              "a fold seen live this session is excluded from the CSV credit (no double count)")

    # profile-derived dataset roster: grid_size()'s "of ~N total" must be a fixed
    # number from minute one (read from the run's own profile), not a guess that
    # grows by a whole dataset's worth of trainings each time
    # 01_meeting01_run_loso.sh's outer loop reaches the next dataset.
    with tempfile.TemporaryDirectory() as td:
        profile_path = os.path.join(td, "roster.json")
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump({"evaluation": {"datasets": ["fsdd", "audiomnist", "mitbih"]}}, fh)
        check(dataset_roster_from_profile(profile_path) == ["fsdd", "audiomnist", "mitbih"],
              "dataset_roster_from_profile reads evaluation.datasets")
        check(dataset_roster_from_profile(os.path.join(td, "missing.json")) is None,
              "dataset_roster_from_profile is None for a missing file (heuristic fallback)")
        bad_path = os.path.join(td, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write("not json")
        check(dataset_roster_from_profile(bad_path) is None,
              "dataset_roster_from_profile is None for malformed JSON")

    check(default_profile_path("meeting01_loso") ==
          os.path.join("src", "experiments", "meeting01", "profiles", "meeting01-loso.json"),
          "default_profile_path maps run_tag underscores to a dashed filename")

    st5 = SessionState()
    st5.apply(ev(type="session_begin", dataset="fsdd", fold=0, seed=42, repeats=2,
                cv_num_folds=6, all_datasets=["fsdd"],
                search_space={"snn_architectures": ["dense"], "ga_population_size": 1,
                              "ga_generations": 0, "encodings": ["direct"],
                              "baselines": ["lstm-ae"]}))
    check(st5.grid_size() == st5.per_fold_trainings() * 6 * 1,
          "without a roster, grid_size() falls back to datasets seen so far (1 so far)")
    st5.set_dataset_roster(["fsdd", "audiomnist", "mitbih"])
    check(st5.grid_size() == st5.per_fold_trainings() * 6 * 3,
          "with a roster, grid_size() uses the full fixed dataset count right away, not a growing guess")

    st6 = SessionState()
    st6.set_dataset_roster(None)
    check(st6.dataset_roster is None, "set_dataset_roster(None) is a no-op")
    st6.set_dataset_roster([])
    check(st6.dataset_roster is None, "set_dataset_roster([]) is also a no-op (falsy list)")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/meeting01")
    ap.add_argument("--run-tag", default="meeting01_loso")
    ap.add_argument("--profile", default=None,
                    help="path to the run's *.json profile, for the FULL dataset list "
                         "(a stable 'of ~N total' denominator). Default: "
                         "src/experiments/meeting01/profiles/<run-tag-with-dashes>.json "
                         "(01_meeting01_run_loso.sh's own convention); silently falls back "
                         "to estimating from datasets seen so far if that file is absent")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between event polls")
    ap.add_argument("--plain", action="store_true", help="periodic text snapshots (no rich)")
    ap.add_argument("--interval", type=float, default=15.0, help="--plain snapshot interval (s)")
    ap.add_argument("--once", action="store_true", help="--plain: print one snapshot and exit")
    ap.add_argument("--rank", type=int, metavar="N",
                    help="print completed config #N's full detail and exit")
    ap.add_argument("--color", action="store_true",
                    help="force ANSI colors in --plain output even when stdout is not a TTY "
                         "(e.g. piping through `less -R`)")
    ap.add_argument("--no-color", action="store_true",
                    help="disable ANSI colors in --plain output (also honors the NO_COLOR "
                         "env var, https://no-color.org)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.color:
        _set_color(True)
    elif args.no_color or os.environ.get("NO_COLOR") is not None:
        _set_color(False)

    if args.self_test:
        _set_color(False)  # deterministic text for the substring checks below
        return _self_test()
    if args.rank is not None:
        return run_detail(args.results_dir, args.run_tag, args.rank)

    if not (args.plain or not sys.stdout.isatty()):
        return run_dashboard(args.results_dir, args.run_tag, args.poll, args.profile)

    tailer = EventTailer(args.results_dir, args.run_tag)
    state = SessionState()
    state.set_dataset_roster(dataset_roster_from_profile(
        args.profile or default_profile_path(args.run_tag)))
    try:
        while True:
            try:
                for ev in tailer.poll():
                    state.apply(ev)
                state.reconcile(tailer.mtimes, tailer.missing)
                state.note_completed_folds(scan_completed_folds(args.results_dir, args.run_tag))
                state.loop_error = ""
            except Exception as exc:  # noqa: BLE001 - a transient FS race must not stop the loop
                state.loop_error = f"{type(exc).__name__}: {exc}"
            if sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")
            print(render_plain(state), flush=True)
            if args.once:
                return 0
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
