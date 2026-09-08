#!/usr/bin/env python3
"""monitor.py — live research dashboard for the nested-LOSO guayaquil run.

The run is 18 independent `guayaquil` processes (one per dataset x outer fold), each
appending structured events to

    results/guayaquil/<run_tag>_<dataset>_fold<f>_events.jsonl

(schema v1, written by GuayaquilEvents.cpp / GuayaquilEventCallback.hpp). This script
tails all of them, folds the events into a session model, and renders a compact
terminal dashboard: session progress, the currently training config(s), live
train/val convergence, a cross-configuration ranking, the hyperparameter search
space with marginal best-val summaries, per-(model, encoding) mean +/- std across
completed seeds x folds, and a scrolling event log. Pressing Enter on a ranking row
opens a detail view with every recorded metric and the reproducibility metadata.

OBSERVABILITY ONLY. This script never writes to the run, never signals it, and is
safe to start, kill, and re-attach at any time.

Modes
  (default)      interactive textual dashboard (requires `textual`; added to
                 scripts/requirements.txt -- rerun `cmake --preset=...` to install)
  --plain        periodic plain-text snapshots; also the automatic mode when stdout
                 is not a TTY (CI / redirect / `| tee`)
  --self-test    synthetic known-answer checks of the aggregator; exits non-zero on
                 failure; needs only the standard library. Safe for CI.

Descriptive only: this tool reports factual training diagnostics (best val loss, the
train/validation gap, epochs without improvement, marginal summaries). It makes no
inferential or causal claim -- no "overfitting", "converged", "significant", or
"generalizes" language.

Usage:
  python3 scripts/pipeline/guayaquil/monitor.py --run-tag article_loso
  python3 scripts/pipeline/guayaquil/monitor.py --results-dir results/guayaquil --plain
  python3 scripts/pipeline/guayaquil/monitor.py --self-test
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
from typing import Any, Iterable, Optional

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
    # metrics keyed by split ("val" / "test")
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_ts: float = 0.0

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
        """The number the ranking sorts on: test-split val loss if evaluated, else the
        best inner-validation loss seen during training."""
        for split in ("test", "val"):
            m = self.metrics.get(split)
            if m and m.get("mse") is not None:
                return m["mse"]
        return self.running_best_val

    def avg_epoch_ms(self) -> Optional[float]:
        return None if not self._epoch_ms else sum(self._epoch_ms) / len(self._epoch_ms)

    _epoch_ms: list[float] = field(default_factory=list)


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
        self.selected: dict[str, dict[str, Any]] = {}  # final_config_id -> selection info
        self.started_wall: Optional[float] = None

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
            c.last_ts = ts
        elif etype == "epoch":
            c = self._config(ev, ds, fold)
            e = int(ev.get("epoch", 0))
            c.epochs.append((e, ev.get("train_loss"), ev.get("val_loss")))
            c.max_epochs = int(ev.get("max_epochs", c.max_epochs) or c.max_epochs)
            ems = ev.get("epoch_ms")
            if isinstance(ems, (int, float)) and math.isfinite(ems):
                c._epoch_ms.append(float(ems))
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

    # ---- mark stalled processes' running configs as failed -----------------------
    def reconcile(self, tailer_mtimes: dict[str, float]) -> None:
        now = time.time()
        for path, proc in self.procs.items():
            mt = tailer_mtimes.get(path, proc.last_ts)
            if proc.is_failed(now, mt):
                proc.error = proc.error or "process stopped without session_end"
                for c in self.configs.values():
                    if c.dataset == proc.dataset and c.fold == proc.fold and c.status == "running":
                        c.status = "failed"

    # ---- derived views ----------------------------------------------------------
    def grid_size(self) -> int:
        s = self.session.get("search_space", {})
        n_snn = (
            len(s.get("snn_architectures", []) or [1])
            * len(s.get("v_th_values", []) or [1])
            * len(s.get("alpha_values", []) or [1])
        )
        per_cell = n_snn + len(s.get("baselines", []) or [])
        n_enc = len(s.get("encodings", []) or [1])
        n_seed = int(self.session.get("repeats", 1) or 1)
        n_fold = int(self.session.get("cv_num_folds", 1) or 1)
        n_ds = len(self.session.get("all_datasets", []) or [1])
        # all_datasets in one process only lists that process's dataset; fall back to a
        # heuristic 3 (fsdd/audiomnist/mitbih) if just one is present but folds imply more.
        if n_ds == 1 and n_fold > 1:
            n_ds = max(1, len({d for d, _ in self.folds_seen})) or 1
        return per_cell * n_enc * n_seed * n_fold * n_ds

    def counts(self) -> dict[str, int]:
        running = done = failed = 0
        for c in self.configs.values():
            if c.role == "snn_sweep":
                # sweep candidates + their config_end (val) — count the val row once
                pass
            if c.status == "running":
                running += 1
            elif c.status == "done":
                done += 1
            elif c.status == "failed":
                failed += 1
        return {"running": running, "done": done, "failed": failed, "total": self.grid_size()}

    def completed_configs(self) -> list[ConfigState]:
        """Headline ranking: baselines + the retrained SNN winner per cell. The
        exploratory `snn_sweep` candidates feed the search-space marginals, not this."""
        return sorted(
            (c for c in self.configs.values()
             if c.status == "done" and c.role != "snn_sweep"),
            key=lambda c: (c.rank_val is None, c.rank_val if c.rank_val is not None else 0.0),
        )

    def active_configs(self) -> list[ConfigState]:
        return sorted(
            (c for c in self.configs.values() if c.status == "running" and c.epochs),
            key=lambda c: -c.last_ts,
        )

    def marginals(self) -> dict[str, list[tuple[str, Optional[float], int]]]:
        """Best inner-val loss grouped by each SNN sweep dimension. Descriptive only."""
        dims: dict[str, dict[str, list[float]]] = {"v_th": {}, "alpha": {}, "architecture": {}}
        for c in self.configs.values():
            if c.role not in ("snn_sweep", "snn_final") or c.status != "done":
                continue
            score = c.rank_val
            if score is None:
                continue
            for key in dims:
                val = c.hyperparams.get(key)
                if val is None:
                    continue
                dims[key].setdefault(str(val), []).append(score)
        out: dict[str, list[tuple[str, Optional[float], int]]] = {}
        for key, buckets in dims.items():
            rows = [(k, min(v) if v else None, len(v)) for k, v in sorted(buckets.items())]
            out[key] = rows
        return out

    def aggregation(self) -> list[tuple[str, str, Optional[float], Optional[float], int, int]]:
        """Per (model, encoding): mean +/- std of best-val over completed (seed x fold),
        plus n_ok / n_failed. Only meaningful with >= 2 observations."""
        groups: dict[tuple[str, str], list[float]] = {}
        failed: dict[tuple[str, str], int] = {}
        for c in self.configs.values():
            if c.role == "snn_sweep":
                continue  # the winner (snn_final) represents the SNN family per cell
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
                if mean is not None and len(vals) >= 2
                else None
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


def _sparkline(values: list[float], width: int = 48) -> str:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return "".join("." for _ in vals)
    blocks = "▁▂▃▄▅▆▇█"
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[min(len(vals) - 1, int(i * step))] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    return "".join(blocks[min(7, int((v - lo) / rng * 7))] for v in vals)


def _ascii_plot(values: list[float], width: int = 48, height: int = 6) -> list[str]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return ["(not enough points)"]
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[min(len(vals) - 1, int(i * step))] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    grid = [[" "] * len(vals) for _ in range(height)]
    for x, v in enumerate(vals):
        y = height - 1 - int((v - lo) / rng * (height - 1))
        grid[y][x] = "*"
    rows = ["".join(r) for r in grid]
    rows[0] = f"{hi:.4f} " + rows[0]
    for i in range(1, len(rows)):
        rows[i] = " " * (len(f"{hi:.4f} ")) + rows[i]
    rows.append(" " * len(f"{hi:.4f} ") + f"{lo:.4f}".ljust(len(vals)))
    return rows


# --------------------------------------------------------------------------------------
# plain-text renderer
# --------------------------------------------------------------------------------------
def render_plain(state: SessionState) -> str:
    ln: list[str] = []
    sess = state.session
    c = state.counts()
    elapsed = time.time() - state.started_wall if state.started_wall else None
    rate = c["done"] / elapsed if elapsed and c["done"] else None
    remaining = (c["total"] - c["done"]) / rate if rate and c["total"] else None
    ln.append(
        f"SESSION {sess.get('run_tag', '?')}  seed={sess.get('seed', '?')}  "
        f"backend={sess.get('backend', '?')}  git={sess.get('git_commit', '?')}"
    )
    ln.append(
        f"  folds seen {len(state.folds_seen)}  configs done {c['done']}  "
        f"running {c['running']}  failed {c['failed']}  of ~{c['total']}  "
        f"elapsed {_hms(elapsed)}  eta {_hms(remaining)}"
    )
    for proc in state.procs.values():
        if proc.error:
            ln.append(f"  FAILED  {proc.dataset} fold{proc.fold}: {proc.error}")

    ln.append("")
    ln.append("ACTIVE")
    for a in state.active_configs()[:6]:
        pct = 100.0 * a.epochs[-1][0] / a.max_epochs if a.max_epochs else 0.0
        ln.append(
            f"  {a.dataset} f{a.fold} {a.model:<14} {a.encoding:<8} {_hp_str(a.hyperparams):<20} "
            f"ep {a.epochs[-1][0]}/{a.max_epochs} {pct:4.0f}%  "
            f"train={_f(a.last_train)}  val={_f(a.last_val)}  "
            f"best_val={_f(a.running_best_val)}@{a.running_best_epoch}  "
            f"gap={_f(a.gap)}  no_improve={a.no_improve}"
        )
        ln.append(f"    train {_sparkline([e[1] for e in a.epochs])}")
        ln.append(f"    val   {_sparkline([e[2] for e in a.epochs])}")
    if not state.active_configs():
        ln.append("  (none training right now)")

    ln.append("")
    ln.append("RANKING (by test/inner-val loss; lower is better)")
    ln.append(
        f"  {'#':>2} {'model':<14} {'enc':<8} {'hp':<20} {'epochs':>6} "
        f"{'val/test':>10} {'mae':>10} {'train_ms':>10} {'params':>9} status"
    )
    for i, cs in enumerate(state.completed_configs()[:12], 1):
        m = cs.metrics.get("test") or cs.metrics.get("val") or {}
        ln.append(
            f"  {i:>2} {cs.model:<14} {cs.encoding:<8} {_hp_str(cs.hyperparams):<20} "
            f"{cs.epochs_run or len(cs.epochs):>6} {_f(cs.rank_val, 6):>10} "
            f"{_f(m.get('mae'), 6):>10} {_f(m.get('train_ms'), 1):>10} "
            f"{str(cs.param_count or '-'):>9} {cs.status}"
        )

    agg = state.aggregation()
    if any(r[4] >= 2 for r in agg):
        ln.append("")
        ln.append("AGGREGATION  best-val mean +/- std over completed seeds x folds")
        for model, enc, mean, std, n_ok, n_fail in agg:
            if n_ok < 2:
                continue
            ln.append(
                f"  {model:<14} {enc:<8}  {_f(mean, 6)} +/- {_f(std, 6)}  "
                f"(n_ok={n_ok} n_failed={n_fail})"
            )

    ln.append("")
    ln.append("RECENT")
    for ev in list(state.events)[-8:]:
        t = time.strftime("%H:%M:%S", time.localtime(float(ev.get("ts_unix", 0))))
        et = ev.get("type", "?")
        if et == "epoch":
            summary = (
                f"{ev.get('config_id', '')} ep {ev.get('epoch')}/{ev.get('max_epochs')} "
                f"train={_f(ev.get('train_loss'))} val={_f(ev.get('val_loss'))}"
            )
        elif et == "config_end":
            mm = ev.get("metrics", {}) or {}
            summary = f"{ev.get('config_id', '')} [{ev.get('split')}] mse={_f(mm.get('mse'))}"
        elif et == "config_selected":
            sel = ev.get("selected", {})
            summary = f"{_hp_str(sel)} val={_f(sel.get('val_score'))}"
        else:
            summary = json.dumps({k: v for k, v in ev.items()
                                  if k not in ("v", "ts_unix", "type", "run_tag")})[:110]
        ln.append(f"  {t}  {et:<15} {summary}")
    return "\n".join(ln)


# --------------------------------------------------------------------------------------
# textual dashboard
# --------------------------------------------------------------------------------------
def run_textual(results_dir: str, run_tag: str, poll: float) -> int:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical, VerticalScroll
        from textual.screen import ModalScreen
        from textual.widgets import DataTable, Footer, Header, Static
    except ModuleNotFoundError:
        print(
            "textual is not installed. Add it to software/nn/scripts/requirements.txt and "
            "rerun `cmake --preset=max-performance`, or use --plain.",
            file=sys.stderr,
        )
        return 2

    tailer = EventTailer(results_dir, run_tag)
    state = SessionState()

    class DetailScreen(ModalScreen):
        BINDINGS = [Binding("escape,q", "dismiss", "Back")]

        def __init__(self, cfg: ConfigState) -> None:
            super().__init__()
            self._cfg = cfg

        def compose(self) -> ComposeResult:
            c = self._cfg
            lines = [
                f"[b]{c.config_id}[/b]",
                f"model={c.model}  encoding={c.encoding}  role={c.role}",
                f"hyperparams: {json.dumps(c.hyperparams)}",
                "",
                "[b]reproducibility[/b]",
                f"seed={c.seed}  fold={c.fold}  run={c.run_id}  dataset={c.dataset}",
                f"config_hash={state.session.get('config_hash')}  "
                f"git_commit={state.session.get('git_commit')}",
                "",
                "[b]training[/b]",
                f"epochs_run={c.epochs_run or len(c.epochs)} / {c.max_epochs}   "
                f"stop_reason={c.stop_reason or '-'}",
                f"best inner-val loss {_f(c.running_best_val)} @ epoch {c.running_best_epoch}",
                f"final train {_f(c.last_train)}   final val {_f(c.last_val)}   "
                f"gap {_f(c.gap)}",
                f"val loss rose for {c.val_increased_epochs} most-recent epoch(s)   "
                f"no improvement for {c.no_improve} epoch(s)",
                f"avg epoch {_f((c.avg_epoch_ms() or 0) / 1000.0, 2)} s",
                "",
                "[b]train / val curve[/b]",
            ]
            lines += _ascii_plot([e[1] for e in c.epochs]) or []
            lines.append("(val)")
            lines += _ascii_plot([e[2] for e in c.epochs]) or []
            lines.append("")
            lines.append("[b]recorded metrics[/b]")
            for split, mm in c.metrics.items():
                lines.append(f"  [{split}]")
                for k, v in mm.items():
                    lines.append(f"    {k:<16} {v}")
            yield VerticalScroll(Static("\n".join(lines)))

    class Monitor(App):
        CSS = """
        Screen { layout: vertical; }
        #session { height: auto; padding: 0 1; background: $panel; }
        #active { height: auto; max-height: 16; padding: 0 1; }
        #mid { height: 1fr; }
        #ranking { width: 2fr; }
        #side { width: 1fr; }
        #log { height: 10; border-top: solid $primary; }
        .hidden { display: none; }
        """
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("enter", "detail", "Detail"),
            Binding("p", "pause", "Pause"),
            Binding("f", "follow", "Follow"),
            Binding("s", "focus('ranking')", "Ranking"),
            Binding("l", "focus('log')", "Log"),
            Binding("question_mark", "help", "Help"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.paused = False
            self.follow = True
            self._narrow = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(id="session")
            yield VerticalScroll(Static(id="active_body"), id="active")
            with Horizontal(id="mid"):
                yield DataTable(id="ranking")
                yield VerticalScroll(Static(id="side_body"), id="side")
            yield Static(id="log")
            yield Footer()

        def on_mount(self) -> None:
            t = self.query_one("#ranking", DataTable)
            t.cursor_type = "row"
            t.add_columns("#", "model", "enc", "hp", "ep", "val", "mae", "train_ms", "params", "st")
            self.set_interval(max(0.2, poll), self.refresh_data)
            self.refresh_data()

        # ---- key actions ---------------------------------------------------------
        def action_pause(self) -> None:
            self.paused = not self.paused

        def action_follow(self) -> None:
            self.follow = not self.follow

        def action_help(self) -> None:
            self.bell()
            self.notify(
                "up/down select - enter detail - p pause - f follow - s ranking - l log - q quit",
                timeout=6,
            )

        def action_detail(self) -> None:
            t = self.query_one("#ranking", DataTable)
            comp = state.completed_configs()
            if 0 <= t.cursor_row < len(comp):
                self.push_screen(DetailScreen(comp[t.cursor_row]))

        # ---- data ---------------------------------------------------------------
        def refresh_data(self) -> None:
            try:
                if not self.paused:
                    for ev in tailer.poll():
                        state.apply(ev)
                    state.reconcile(tailer.mtimes)
                self._render()
            except Exception as exc:  # noqa: BLE001  never let a frame kill the app
                try:
                    self.query_one("#log", Static).update(f"[red]render error:[/red] {exc}")
                except Exception:  # noqa: BLE001
                    pass

        def on_resize(self, event) -> None:  # noqa: ANN001
            self._narrow = event.size.width < 80
            for wid in ("#mid", "#side"):
                try:
                    self.query_one(wid).set_class(self._narrow, "hidden")
                except Exception:  # noqa: BLE001  widgets not mounted yet
                    pass

        def _render(self) -> None:
            sess, c = state.session, state.counts()
            elapsed = time.time() - state.started_wall if state.started_wall else None
            rate = c["done"] / elapsed if elapsed and c["done"] else None
            eta = (c["total"] - c["done"]) / rate if rate and c["total"] else None
            fails = [p for p in state.procs.values() if p.error]
            self.query_one("#session", Static).update(
                f"[b]{sess.get('run_tag', '?')}[/b]  seed {sess.get('seed', '?')}  "
                f"backend {sess.get('backend', '?')}  git {sess.get('git_commit', '?')}   "
                f"folds {len(state.folds_seen)}   done [green]{c['done']}[/green]  "
                f"running [cyan]{c['running']}[/cyan]  failed [red]{c['failed']}[/red]  "
                f"of ~{c['total']}   elapsed {_hms(elapsed)}   eta {_hms(eta)}"
                + (f"\n[red]failed:[/red] "
                   + "; ".join(f"{p.dataset} f{p.fold}: {p.error}" for p in fails) if fails else "")
            )

            act = state.active_configs()
            if self.follow and self._narrow:
                act = act[:1]
            body = []
            for a in act[:6]:
                ep = a.epochs[-1][0]
                pct = 100.0 * ep / a.max_epochs if a.max_epochs else 0.0
                body.append(
                    f"[b cyan]{a.dataset} f{a.fold}[/b cyan] {a.model} {a.encoding} "
                    f"{_hp_str(a.hyperparams)}  ep {ep}/{a.max_epochs} {pct:.0f}%"
                )
                body.append(
                    f"  train {_f(a.last_train)}  val {_f(a.last_val)}  "
                    f"best_val {_f(a.running_best_val)}@{a.running_best_epoch}  "
                    f"gap {_f(a.gap)}  no_improve {a.no_improve}  "
                    f"val_rose {a.val_increased_epochs}ep"
                )
                body.append(f"  T {_sparkline([e[1] for e in a.epochs])}")
                body.append(f"  V {_sparkline([e[2] for e in a.epochs])}")
            self.query_one("#active_body", Static).update("\n".join(body) or "(nothing training)")

            t = self.query_one("#ranking", DataTable)
            keep = t.cursor_row
            t.clear()
            for i, cs in enumerate(state.completed_configs()[:200], 1):
                m = cs.metrics.get("test") or cs.metrics.get("val") or {}
                t.add_row(
                    str(i), cs.model, cs.encoding, _hp_str(cs.hyperparams),
                    str(cs.epochs_run or len(cs.epochs)), _f(cs.rank_val, 5),
                    _f(m.get("mae"), 5), _f(m.get("train_ms"), 0),
                    str(cs.param_count or "-"), cs.status,
                )
            if keep is not None:
                try:
                    t.move_cursor(row=min(keep, t.row_count - 1))
                except Exception:  # noqa: BLE001
                    pass

            side = ["[b]search space[/b]"]
            s = sess.get("search_space", {})
            for k in ("snn_architectures", "v_th_values", "alpha_values", "encodings", "baselines"):
                if s.get(k):
                    side.append(f"  {k}: {', '.join(str(x) for x in s[k])}")
            grid = state.grid_size()
            side.append(f"  grid ~ {grid} configs   done {c['done']}")
            side.append("")
            side.append("[b]marginal best inner-val (descriptive, not causal)[/b]")
            for dim, rows in state.marginals().items():
                for label, best, n in rows:
                    side.append(f"  {dim}={label:<12} {_f(best)}  (n={n})")
            agg = state.aggregation()
            if any(r[4] >= 2 for r in agg):
                side.append("")
                side.append("[b]best-val mean +/- std (completed seeds x folds)[/b]")
                for model, enc, mean, std, n_ok, n_fail in agg:
                    if n_ok >= 2:
                        side.append(
                            f"  {model} {enc}: {_f(mean, 5)} +/- {_f(std, 5)} "
                            f"(n_ok={n_ok} n_failed={n_fail})"
                        )
            self.query_one("#side_body", Static).update("\n".join(side))

            log = []
            for ev in list(state.events)[-40:]:
                tt = time.strftime("%H:%M:%S", time.localtime(float(ev.get("ts_unix", 0))))
                et = ev.get("type", "?")
                colour = {
                    "session_error": "red", "config_end": "green",
                    "config_selected": "yellow", "fold_begin": "cyan",
                }.get(et, "white")
                if et == "epoch":
                    txt = (f"{ev.get('config_id', '')} ep {ev.get('epoch')} "
                           f"train={_f(ev.get('train_loss'))} val={_f(ev.get('val_loss'))}")
                elif et == "config_end":
                    mm = ev.get("metrics", {}) or {}
                    txt = f"{ev.get('config_id', '')} [{ev.get('split')}] mse={_f(mm.get('mse'))}"
                else:
                    txt = json.dumps({k: v for k, v in ev.items()
                                      if k not in ("v", "ts_unix", "type", "run_tag", "dataset",
                                                   "fold")})[:120]
                log.append(f"[dim]{tt}[/dim] [{colour}]{et:<14}[/{colour}] {txt}")
            self.query_one("#log", Static).update("\n".join(log[-9:]))

    return Monitor().run() or 0


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
    # a baseline that improves then plateaus
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
    check(c.param_count == 1234, "param_count carried from config_end metrics")
    check(st.counts()["done"] == 1 and st.counts()["running"] == 0, "counts: 1 done, 0 running")

    # second seed -> aggregation mean/std over 2 obs
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

    # failed process detection
    st.apply(ev(type="config_begin", dataset="fsdd", fold=1, config_id="gru-ae_direct_seed42_run1",
               model="gru-ae", encoding="direct", role="baseline", run_id=1, seed=42,
               max_epochs=10, _path="/tmp/x_fsdd_fold1_events.jsonl"))
    st.apply(ev(type="session_error", dataset="fsdd", fold=1, what="boom",
               _path="/tmp/x_fsdd_fold1_events.jsonl"))
    st.reconcile({})
    check(st.configs["gru-ae_direct_seed42_run1"].status == "failed",
          "running config of an errored process -> failed")
    check(st.counts()["failed"] == 1, "counts: 1 failed")

    # missing/unknown tolerance
    st.apply(ev(type="epoch", dataset="fsdd", fold=0, config_id="lstm-ae_direct_seed42_run1",
               epoch=6, max_epochs=10, train_loss=None, val_loss=None))
    st.apply(ev(type="totally_unknown_event", dataset="fsdd", fold=0))
    st.apply({"type": "epoch", "v": 999})  # wrong schema version -> ignored
    check(True, "unknown event type / null metrics / wrong version tolerated (no exception)")

    # renders without raising
    txt = render_plain(st)
    banned = ["overfit", "converged", "convergence achieved", "significant", "generalizes",
              "generalisation is", "the model is better"]
    check(all(b not in txt.lower() for b in banned), "plain render has no inferential language")
    check("RANKING" in txt and "ACTIVE" in txt, "plain render has the core sections")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/guayaquil")
    ap.add_argument("--run-tag", default="article_loso")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between event polls")
    ap.add_argument("--plain", action="store_true", help="periodic text snapshots (no textual)")
    ap.add_argument("--interval", type=float, default=15.0, help="--plain snapshot interval (s)")
    ap.add_argument("--once", action="store_true", help="--plain: print one snapshot and exit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    plain = args.plain or not sys.stdout.isatty()
    if not plain:
        return run_textual(args.results_dir, args.run_tag, args.poll)

    tailer = EventTailer(args.results_dir, args.run_tag)
    state = SessionState()
    try:
        while True:
            for ev in tailer.poll():
                state.apply(ev)
            state.reconcile(tailer.mtimes)
            if sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")  # only when a real terminal
            print(render_plain(state), flush=True)
            if args.once:
                return 0
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
