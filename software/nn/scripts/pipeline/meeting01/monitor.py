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
                 TTY (CI / redirect / `| tee`).  --once prints one snapshot.
  --rank N       print the full detail of completed config #N (as ranked) and exit.
  --self-test    synthetic known-answer checks of the aggregator; stdlib only; CI-safe.

Descriptive only -- factual training diagnostics, no inferential or causal language
("overfitting", "converged", "significant", "generalizes").
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
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

    def poll(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(glob.glob(self._pattern)):
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

    def reconcile(self, tailer_mtimes: dict[str, float]) -> None:
        now = time.time()
        self._mtimes = dict(tailer_mtimes)
        for path, proc in self.procs.items():
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
        s = self.session.get("search_space", {})
        n_snn = (
            len(s.get("snn_architectures", []) or [1])
            * len(s.get("v_th_values", []) or [1])
            * len(s.get("alpha_values", []) or [1])
        )
        per_cell = n_snn + len(s.get("baselines", []) or [])
        return per_cell * len(s.get("encodings", []) or [1]) * int(self.session.get("repeats", 1) or 1)

    def grid_size(self) -> int:
        n_fold = int(self.session.get("cv_num_folds", 1) or 1)
        n_ds = len(self.session.get("all_datasets", []) or [1])
        if n_ds == 1 and n_fold > 1:
            n_ds = max(1, len({d for d, _ in self.folds_seen}))
        return self.per_fold_trainings() * n_fold * n_ds

    def counts(self) -> dict[str, int]:
        running = done = failed = 0
        for c in self.configs.values():
            if c.status == "running":
                running += 1
            elif c.status == "done":
                done += 1
            elif c.status == "failed":
                failed += 1
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
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return ["(not enough points yet)"]
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[min(len(vals) - 1, int(i * step))] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    grid = [[" "] * len(vals) for _ in range(height)]
    for x, v in enumerate(vals):
        y = height - 1 - int((v - lo) / rng * (height - 1))
        grid[y][x] = "*"
    pad = len(f"{hi:.4f} ")
    rows = [f"{hi:.4f} " + "".join(grid[0])]
    rows += [" " * pad + "".join(r) for r in grid[1:]]
    rows.append(f"{lo:.4f} ".rjust(pad) + " " * len(vals))
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
    ln.append(f"SESSION  {sess.get('run_tag', '?')}  seed {sess.get('seed', '?')}  "
              f"backend {sess.get('backend', '?')}  git {sess.get('git_commit', '?')}")
    ln.append(f"  done {c['done']}   running {c['running']}   failed {c['failed']}   "
              f"of ~{c['total']}   elapsed {_hms(elapsed)}   eta {_hms(state.eta_seconds())} (rough)")
    for proc in state.procs.values():
        if proc.error:
            ln.append(f"  FAILED  {proc.dataset} fold{proc.fold}: {proc.error}")
    if state.loop_error:
        ln.append(f"  poll error (retrying): {state.loop_error}")

    ln.append("")
    ln.append("TRAINING NOW")
    act = state.active_configs()
    if not act:
        live = state.live_procs()
        if live:
            where = ", ".join(f"{p.dataset} fold {p.fold}" for p in live)
            ln.append(f"  process alive ({where}) — starting the next config "
                      "(no epoch reported yet)")
        else:
            ln.append("  nothing training right now (between folds / aggregating / not started)")
    for a in act[:6]:
        done_ep = len(a.epochs)
        run_ep = a.cur_progress_epoch or (done_ep + 1)
        ep_frac = done_ep / a.max_epochs if a.max_epochs else 0.0
        ln.append(f"  {a.dataset} fold {a.fold}  {a.model} {a.encoding} "
                  f"{_hp_inline(a.hyperparams)}".rstrip())
        ln.append(f"    epoch {run_ep}/{a.max_epochs} [{_bar(ep_frac)}] {done_ep} done")
        if a.cur_total_batches > 1:
            eta_e = _hms(a.cur_epoch_eta_s) if (a.cur_epoch_eta_s or 0) > 0 else "?"
            bl = f"   loss {_f(a.cur_batch_loss, 5)}" if a.cur_batch_loss is not None else ""
            ln.append(f"    batch {a.cur_batch}/{a.cur_total_batches} "
                      f"[{_bar(a.cur_batch_frac)}] {a.cur_batch_frac * 100:.0f}%{bl}"
                      f"   ~{eta_e} left this epoch")
        ln.append(f"    train {_f(a.last_train)}   val {_f(a.last_val)}   "
                  f"best val {_f(a.running_best_val)} @ ep {a.running_best_epoch}   "
                  f"gap(val-train) {_f(a.gap, 5)}   no improvement {a.no_improve} ep")
        ln.append(f"    train {_spark([e[1] for e in a.epochs])}")
        ln.append(f"    val   {_spark([e[2] for e in a.epochs])}")

    ln.append("")
    ln.append("COMPLETED  (ranked by held-out test loss, else best inner-validation loss)")
    comp = state.completed_configs()
    if not comp:
        ln.append("  nothing finished yet - the first config takes ~10-20 min after a fold starts")
    else:
        ln.append(f"  {'#':>2}  {'model':<14} {'enc':<8} {'hp':<18} {'epochs':>6} "
                  f"{'loss':>10} {'mae':>10} {'train s':>8} {'params':>9}")
        for i, cs in enumerate(comp[:14], 1):
            m = cs.metrics.get("test") or cs.metrics.get("val") or {}
            tms = m.get("train_ms")
            ln.append(f"  {i:>2}  {cs.model:<14} {cs.encoding:<8} {_hp_str(cs.hyperparams):<18} "
                      f"{cs.epochs_run or len(cs.epochs):>6} {_f(cs.rank_val, 6):>10} "
                      f"{_f(m.get('mae'), 6):>10} "
                      f"{(f'{tms / 1000:.0f}' if tms else '-'):>8} {str(cs.param_count or '-'):>9}")

    agg = [r for r in state.aggregation() if r[4] >= 2]
    if agg:
        ln.append("")
        ln.append("MEAN +/- STD  of loss over completed seeds x folds")
        for model, enc, mean, std, n_ok, n_fail in agg:
            ln.append(f"  {model:<14} {enc:<8}  {_f(mean, 6)} +/- {_f(std, 6)}  "
                      f"(n={n_ok}{f', {n_fail} failed' if n_fail else ''})")

    ln.append("")
    ln.append("RECENT")
    for ev in list(state.events)[-8:]:
        t, et, s = _event_line(ev)
        ln.append(f"  {t}  {et:<15} {s}")
    return "\n".join(ln)


# --------------------------------------------------------------------------------------
# rich renderers
# --------------------------------------------------------------------------------------
def _panel_session(state: SessionState):  # noqa: ANN201
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
        g.add_row(Text("waiting for the first event... start the run with:", style="yellow"))
        g.add_row(Text("  EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/"
                       "01_meeting01_run_loso.sh", style="dim"))
    else:
        g.add_row(Text.assemble(
            (str(sess.get("run_tag", "?")), "bold"),
            (f"   seed {sess.get('seed', '?')}   backend {sess.get('backend', '?')}"
             f"   git {sess.get('git_commit', '?')}", "dim"),
        ))
        g.add_row(Text.assemble(
            (f"[{_bar(frac, 26)}] ", "cyan"),
            (f"{c['done']} done", "green"),
            ("  /  ", "dim"),
            (f"{c['running']} running", "cyan"),
            ("  /  ", "dim"),
            (f"{c['failed']} failed", "red" if c["failed"] else "dim"),
            (f"   of ~{c['total']} trainings", "dim"),
        ))
        g.add_row(Text(f"elapsed {_hms(elapsed)}   eta {_hms(state.eta_seconds())} (rough, "
                       f"from completed-so-far rate)", style="dim"))
        sp = sess.get("search_space", {})
        n_arch = len(sp.get("snn_architectures", []) or [])
        n_v = len(sp.get("v_th_values", []) or [])
        n_a = len(sp.get("alpha_values", []) or [])
        n_e = len(sp.get("encodings", []) or [])
        n_b = len(sp.get("baselines", []) or [])
        g.add_row(Text(
            f"grid: {n_b} baselines + {n_arch}×{n_v}×{n_a} SNN sweep + retrain, "
            f"× {n_e} encodings × {sess.get('repeats', '?')} seeds  "
            f"≈ {state.per_fold_trainings()}/fold", style="dim"))
        fails = [p for p in state.procs.values() if p.error]
        if fails:
            g.add_row(Text("  ".join(f"[FAILED {p.dataset} f{p.fold}] {p.error}" for p in fails),
                           style="red"))
    return Panel(g, title="SESSION", title_align="left", border_style="blue", padding=(0, 1))


def _panel_now(state: SessionState):  # noqa: ANN201
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    act = state.active_configs()
    if not act:
        live = state.live_procs()
        if live:
            where = ", ".join(f"{p.dataset} fold {p.fold}" for p in live)
            body = Text(f"Process alive ({where}) — starting the next config.\n"
                        "The fast SNN sweep briefly shows this between configs; "
                        "an epoch will appear within a few seconds.", style="cyan")
        else:
            body = Text("Nothing is training right now.\n"
                        "The run may be between folds, running the Python aggregation, "
                        "or not started yet.", style="yellow")
        return Panel(body, title="TRAINING NOW", title_align="left",
                     border_style="grey50", padding=(0, 1))

    blocks = []
    for a in act[:3]:
        done_ep = len(a.epochs)
        run_ep = a.cur_progress_epoch or (done_ep + 1)
        ep_frac = done_ep / a.max_epochs if a.max_epochs else 0.0
        avg = a.avg_epoch_ms()
        eta_tr = (_hms((a.max_epochs - done_ep) * avg / 1000.0)
                  if avg and a.max_epochs else None)
        t = Table.grid(padding=(0, 1))
        t.add_column()
        hp = _hp_inline(a.hyperparams)
        t.add_row(Text.assemble(
            (f"{a.dataset} fold {a.fold}", "bold cyan"),
            (f"   {a.model}  {a.encoding}{('  ' + hp) if hp else ''}", "bold"),
        ))
        t.add_row(Text.assemble(
            (f"epoch {run_ep}/{a.max_epochs}  ", ""),
            (f"[{_bar(ep_frac, 22)}] ", "cyan"),
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
            (f"best val {_f(a.running_best_val)} @ ep {a.running_best_epoch}", "green"),
        ))
        t.add_row(Text(f"gap (val - train) {_f(a.gap, 5)}    "
                       f"no improvement for {a.no_improve} epoch(s)    "
                       f"val loss rose {a.val_increased_epochs} of the last epochs", style="dim"))
        hint = "" if len(a.epochs) >= 3 else "   (fills in as epochs complete)"
        t.add_row(Text.assemble(("train  ", "dim"),
                                (_spark([e[1] for e in a.epochs]) or "·", "white"),
                                (hint, "dim")))
        t.add_row(Text.assemble(("val    ", "dim"),
                                (_spark([e[2] for e in a.epochs]) or "·", "white")))
        blocks.append(t)
    return Panel(Group(*blocks), title="TRAINING NOW", title_align="left",
                 border_style="cyan", padding=(0, 1))


def _panel_ranking(state: SessionState, max_rows: int = 12):  # noqa: ANN201
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    comp = state.completed_configs()
    title = "COMPLETED  —  ranked by held-out test loss (else best inner-validation loss)"
    if not comp:
        body = Text("No config has finished yet.\n"
                    "Per fold: 3 baselines + a 27-config SNN v_th×α×arch sweep + "
                    "1 retrain,  × 3 encodings × 5 seeds.\n"
                    "The first (LSTM-AE) result lands ~10–20 min after a fold starts.",
                    style="yellow")
        return Panel(body, title=title, title_align="left", border_style="grey50", padding=(0, 1))

    tbl = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, header_style="bold")
    for name, just in (("#", "right"), ("model", "left"), ("enc", "left"), ("hp", "left"),
                       ("ep", "right"), ("loss", "right"), ("kind", "left"),
                       ("mae", "right"), ("train s", "right"), ("params", "right")):
        tbl.add_column(name, justify=just, no_wrap=True)
    best_loss = comp[0].rank_val
    for i, cs in enumerate(comp[:max_rows], 1):
        m = cs.metrics.get("test") or cs.metrics.get("val") or {}
        tms = m.get("train_ms")
        style = "green" if cs.rank_val == best_loss else ""
        tbl.add_row(
            str(i), cs.model, cs.encoding, _hp_str(cs.hyperparams),
            str(cs.epochs_run or len(cs.epochs)),
            _f(cs.rank_val, 6), cs.rank_val_kind,
            _f(m.get("mae"), 6),
            f"{tms / 1000:.0f}" if tms else "-",
            str(cs.param_count or "-"),
            style=style,
        )
    extra = "" if len(comp) <= max_rows else f"   (+{len(comp) - max_rows} more — --rank N for detail)"
    parts = [tbl, Text(f"monitor.py --rank N  for one row's full detail{extra}", style="dim")]

    marg = state.marginals()
    if any(any(n for _, _, n in rs) for rs in marg.values()):
        mt = Table.grid(padding=(0, 2))
        mt.add_column(style="dim")
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

    return Panel(Group(*parts), title=title, title_align="left", border_style="blue",
                 padding=(0, 1))


def _panel_events(state: SessionState, n: int = 8):  # noqa: ANN201
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    colour = {"session_error": "red", "config_end": "green", "config_selected": "yellow",
              "fold_begin": "cyan", "fold_end": "cyan", "train_end": "green",
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
    return Panel(t, title="RECENT", title_align="left", border_style="blue", padding=(0, 1))


def render_dashboard(state: SessionState, height: int = 40):  # noqa: ANN201
    """Full-terminal layout: SESSION and TRAINING NOW at fixed heights, COMPLETED
    takes the slack, RECENT pinned to the bottom. Each region clips its content, so
    a short terminal just shows fewer ranking / event rows — nothing overflows."""
    from rich.layout import Layout
    from rich.panel import Panel

    def _safe(fn, *a):  # noqa: ANN001
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001
            return Panel(f"[red]panel error:[/red] {exc}", border_style="red")

    ev_rows = max(3, min(9, height - 26))
    rank_rows = max(3, height - 12 - ev_rows - 12)

    root = Layout()
    root.split_column(
        Layout(_safe(_panel_session, state), name="session", size=7),
        Layout(_safe(_panel_now, state), name="now", size=11),
        Layout(_safe(_panel_ranking, state, rank_rows), name="done", ratio=1, minimum_size=5),
        Layout(_safe(_panel_events, state, ev_rows), name="recent", size=ev_rows + 2),
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


def _drain(results_dir: str, run_tag: str) -> SessionState:
    tailer = EventTailer(results_dir, run_tag)
    state = SessionState()
    for ev in tailer.poll():
        state.apply(ev)
    state.reconcile(tailer.mtimes)
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


def run_dashboard(results_dir: str, run_tag: str, poll: float) -> int:
    try:
        from rich.console import Console
        from rich.live import Live
    except ModuleNotFoundError:
        print("rich is not installed. It is in scripts/requirements.txt — rerun "
              "`cmake --preset=max-performance`, or use --plain.", file=sys.stderr)
        return 2

    tailer = EventTailer(results_dir, run_tag)
    state = SessionState()
    console = Console()
    try:
        with Live(render_dashboard(state, console.size.height), console=console, screen=True,
                  refresh_per_second=4, redirect_stderr=False) as live:
            consecutive_errors = 0
            while True:
                try:
                    for ev in tailer.poll():
                        state.apply(ev)
                    state.reconcile(tailer.mtimes)
                    live.update(render_dashboard(state, console.size.height))
                    consecutive_errors = 0
                except Exception as exc:  # noqa: BLE001
                    # A transient FS race (a results file swapped/removed by a
                    # concurrent git op, a half-written line) must not freeze the
                    # dashboard. Keep polling; surface it, bail only if it never clears.
                    consecutive_errors += 1
                    state.loop_error = f"{type(exc).__name__}: {exc}"
                    if consecutive_errors >= 30:
                        raise
                time.sleep(max(0.5, poll))
    except KeyboardInterrupt:
        return 0


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------
def _self_test() -> int:
    def ev(**kw: Any) -> dict[str, Any]:
        kw.setdefault("v", 1)
        kw.setdefault("ts_unix", time.time())
        kw.setdefault("run_tag", "t")
        return kw

    st = SessionState()
    st.apply(ev(type="session_begin", dataset="fsdd", fold=0, seed=42, repeats=2,
               cv_num_folds=6, all_datasets=["fsdd"],
               search_space={"snn_architectures": ["dense"], "v_th_values": [1.0],
                             "alpha_values": [0.9], "encodings": ["direct"],
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

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/meeting01")
    ap.add_argument("--run-tag", default="meeting01_loso")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between event polls")
    ap.add_argument("--plain", action="store_true", help="periodic text snapshots (no rich)")
    ap.add_argument("--interval", type=float, default=15.0, help="--plain snapshot interval (s)")
    ap.add_argument("--once", action="store_true", help="--plain: print one snapshot and exit")
    ap.add_argument("--rank", type=int, metavar="N",
                    help="print completed config #N's full detail and exit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.rank is not None:
        return run_detail(args.results_dir, args.run_tag, args.rank)

    if not (args.plain or not sys.stdout.isatty()):
        return run_dashboard(args.results_dir, args.run_tag, args.poll)

    tailer = EventTailer(args.results_dir, args.run_tag)
    state = SessionState()
    try:
        while True:
            try:
                for ev in tailer.poll():
                    state.apply(ev)
                state.reconcile(tailer.mtimes)
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
