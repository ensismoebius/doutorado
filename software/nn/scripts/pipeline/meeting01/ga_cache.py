#!/usr/bin/env python3
"""ga_cache.py — incremental tailer + aggregation for GA per-individual cache files.

Mirrors monitor.py's EventTailer design: byte-offset incremental reads,
truncation-tolerant, new-file-picking-up.  Each cache file is a JSONL of
individual_to_checkpoint_json() outputs (Meeting01GaCheckpoint.hpp), one per
line, flushed the instant a genome is scored.

Identity (dataset, fold, run_id, family) is encoded only in the FILENAME, not
the JSON payload — the per-cell composite run_tag is:

    <global_run_tag>_<dataset>_fold<f>_run<r>[_<family>]

where <family> is omitted for SNN (the default arm) and is _lstm, _gru, or
_transformer for the three baselines.

Cache files are resume ARTIFACTS: deleted on successful completion
(remove_checkpoint_artifacts).  A vanished file means "this family's search
for this cell finished", not an error.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from typing import Any, Optional

# Filename pattern: meeting01_ga_<run_tag>_<dataset>_fold<f>_run<r>[_<family>]_cache.jsonl
# The run_tag passed to checkpoint_cache_path is the per-cell composite tag.
_CACHE_RE = re.compile(
    r"meeting01_ga_(?P<run_tag>.+?)_(?P<ds>[a-z0-9]+)_fold(?P<fold>\d+)"
    r"_run(?P<run>\d+)(?:_(?P<family>lstm|gru|transformer))?_cache\.jsonl$"
)

_FAMILY_SNN = "snn"


class GaCacheTailer:
    """Follows every matching *_cache.jsonl, yielding parsed individual dicts
    annotated with identity extracted from the filename.

    Tracks a byte offset per file; on each poll it reads only appended bytes
    and parses whole lines (a trailing partial line is retried next poll).  New
    files appearing in the directory are picked up automatically.
    """

    def __init__(self, results_dir: str, run_tag: str) -> None:
        self._results_dir = results_dir
        self._run_tag = run_tag
        self._pattern = os.path.join(
            results_dir, f"meeting01_ga_{glob.escape(run_tag)}_*_cache.jsonl"
        )
        self._offsets: dict[str, int] = {}
        self._carry: dict[str, str] = {}
        self.mtimes: dict[str, float] = {}
        self.missing: set[str] = set()

    def poll(self) -> list[dict[str, Any]]:
        """Return newly-read individual dicts since last poll.

        Each dict carries the original JSONL fields plus:
          _path, _dataset, _fold, _run_id, _family
        """
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
            if size < start:
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
            self._carry[path] = lines.pop()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ind = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ind, dict):
                    continue
                identity = _parse_identity(path)
                if identity:
                    ind["_path"] = path
                    ind["_dataset"] = identity["dataset"]
                    ind["_fold"] = identity["fold"]
                    ind["_run_id"] = identity["run_id"]
                    ind["_family"] = identity["family"]
                out.append(ind)
        return out

    def active_cells(self) -> set[tuple[str, int, int, str]]:
        """Return the set of (dataset, fold, run_id, family) cells that
        currently have a cache file on disk — i.e. a sweep is in flight."""
        cells: set[tuple[str, int, int, str]] = set()
        for path in glob.glob(self._pattern):
            identity = _parse_identity(path)
            if identity:
                cells.add((
                    identity["dataset"],
                    identity["fold"],
                    identity["run_id"],
                    identity["family"],
                ))
        return cells


def _parse_identity(path: str) -> Optional[dict[str, Any]]:
    """Extract dataset/fold/run_id/family from a cache filename."""
    m = _CACHE_RE.search(os.path.basename(path))
    if not m:
        return None
    return {
        "dataset": m["ds"],
        "fold": int(m["fold"]),
        "run_id": int(m["run"]),
        "family": m["family"] or _FAMILY_SNN,
    }


# ── Per-cell aggregation ─────────────────────────────────────────────────────


class GaSearchState:
    """Aggregated view of one (dataset, fold, run_id, family) GA search.

    Buckets individuals by born_generation and maintains the Pareto frontier.
    """

    def __init__(
        self, dataset: str, fold: int, run_id: int, family: str
    ) -> None:
        self.dataset = dataset
        self.fold = fold
        self.run_id = run_id
        self.family = family
        self.individuals: list[dict[str, Any]] = []
        self._dirty = True
        self._pareto: list[dict[str, Any]] = []
        self._by_generation: dict[int, list[dict[str, Any]]] = {}

    def add(self, ind: dict[str, Any]) -> None:
        self.individuals.append(ind)
        self._dirty = True

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self._by_generation.clear()
        for ind in self.individuals:
            gen = ind.get("born_generation", 0)
            self._by_generation.setdefault(gen, []).append(ind)
        self._pareto = pareto_frontier(self.individuals)

    @property
    def pareto(self) -> list[dict[str, Any]]:
        self._rebuild()
        return self._pareto

    @property
    def by_generation(self) -> dict[int, list[dict[str, Any]]]:
        self._rebuild()
        return self._by_generation

    @property
    def n_individuals(self) -> int:
        return len(self.individuals)

    @property
    def generations(self) -> list[int]:
        self._rebuild()
        return sorted(self._by_generation.keys())

    @property
    def best_val_mse(self) -> Optional[float]:
        vals = [v for v in (ind.get("val_mse") for ind in self.individuals
                            if ind.get("feasible", True))
                if v is not None and isinstance(v, (int, float))]
        return min(vals) if vals else None

    @property
    def best_inference_cost(self) -> Optional[int]:
        costs = [int(v) for v in (ind.get("inference_cost") for ind in self.individuals
                                  if ind.get("feasible", True))
                 if v is not None and isinstance(v, (int, float))]
        return min(costs) if costs else None

    def summary(self) -> dict[str, Any]:
        """JSON-serializable progress summary for this cell."""
        self._rebuild()
        return {
            "dataset": self.dataset,
            "fold": self.fold,
            "run_id": self.run_id,
            "family": self.family,
            "n_individuals": self.n_individuals,
            "generations": self.generations,
            "max_generation": max(self.generations) if self.generations else 0,
            "best_val_mse": _finite_or_none(self.best_val_mse),
            "best_inference_cost": self.best_inference_cost,
            "pareto": [_finite_or_none(ind) for ind in self._pareto],
            "mutation_count": None,
            "crossover_count": None,
            "mutation_crossover_note": (
                "Not available — mutation/crossover counts are not recorded "
                "in the GA cache file (individual_to_checkpoint_json outputs "
                "genome, val_mse, param_count, inference_cost, feasible, "
                "constraint_violation, objectives, born_generation only)."
            ),
        }


# ── Pareto frontier ──────────────────────────────────────────────────────────


def pareto_frontier(
    individuals: list[dict[str, Any]],
    objectives: tuple[str, str] = ("val_mse", "inference_cost"),
) -> list[dict[str, Any]]:
    """Feasibility-first 2-objective non-dominated sort.

    Returns the non-dominated (Pareto-optimal) subset of *individuals*.
    Feasible individuals always dominate infeasible ones.  Ties on both
    objectives are kept (both are Pareto-optimal).  Matches the rule from
    Nsga2Core.hpp's feasibility-first sorting.

    Parameters
    ----------
    individuals : list[dict]
        Each dict must have keys *objectives[0]* and *objectives[1]* (numeric)
        and "feasible" (bool).
    objectives : tuple[str, str]
        The two objective keys to optimize (minimize).  Default:
        ("val_mse", "inference_cost").
    """
    obj_min, obj_max = objectives

    feasible = [ind for ind in individuals if ind.get("feasible", True)]
    infeasible = [ind for ind in individuals if not ind.get("feasible", True)]

    # Among feasible: standard non-dominated sort (minimize both)
    front: list[dict[str, Any]] = []
    for i, a in enumerate(feasible):
        a_min = a.get(obj_min)
        a_max = a.get(obj_max)
        if a_min is None or a_max is None:
            continue
        dominated = False
        for j, b in enumerate(feasible):
            if i == j:
                continue
            b_min = b.get(obj_min)
            b_max = b.get(obj_max)
            if b_min is None or b_max is None:
                continue
            # b dominates a if b <= a on all objectives and b < a on at least one
            if (b_min <= a_min and b_max <= a_max
                    and (b_min < a_min or b_max < a_max)):
                dominated = True
                break
        if not dominated:
            front.append(a)

    # Infeasible individuals that would otherwise be on the front are included
    # only if they are not dominated by any OTHER infeasible individual — but
    # they are always dominated by any feasible one, so they only appear if
    # there are no feasible individuals at all.
    if not front and infeasible:
        for i, a in enumerate(infeasible):
            a_cv = a.get("constraint_violation", float("inf"))
            dominated = False
            for j, b in enumerate(infeasible):
                if i == j:
                    continue
                b_cv = b.get("constraint_violation", float("inf"))
                if b_cv <= a_cv and b_cv < a_cv:
                    dominated = True
                    break
            if not dominated:
                front.append(a)

    return front


# ── Helpers ──────────────────────────────────────────────────────────────────

def _finite_or_none(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite_or_none(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite_or_none(x) for x in v]
    return v
