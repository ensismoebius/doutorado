#!/usr/bin/env python3
"""sweep_state.py — GA-sweep identity uncertainty annotation.

Because EventContext is a single global slot (Meeting01Events.hpp:127) that is
only updated via set_pending_context() at two call sites (Meeting01Experiment.cpp
487,692) — both after a family's entire NSGA-II search finishes — every
individual's config_begin/epoch_progress/epoch/train_end events during a sweep
carry stale identity from whatever the previous final retrain set.

The dashboard must NOT present per-genome epoch/loss claims as reliable during
a sweep.  This module annotates active configs with `sweep_uncertain: bool` so
the frontend can grey out or replace per-genome identity claims.

Mitigation: a cache file currently existing for (dataset,fold) is an
unambiguous, C++-unchanged signal that a sweep is in flight for that cell.
"""
from __future__ import annotations

from typing import Any


def annotate_sweep_uncertain(
    active_configs: list[dict[str, Any]],
    active_ga_cells: set[tuple[str, int, int, str]],
) -> list[dict[str, Any]]:
    """Tag each active config dict with `sweep_uncertain`.

    A config is sweep-uncertain if ANY GA cache file exists for its
    (dataset, fold) — because the four family searches run strictly
    sequentially within one process, "a cache file currently exists" is
    an unambiguous signal that a sweep is in flight.

    Parameters
    ----------
    active_configs : list[dict]
        Output of SessionState.active_configs_json() — each dict has
        "dataset", "fold", and other config fields.
    active_ga_cells : set[tuple[str, int, int, str]]
        Output of GaCacheTailer.active_cells() — set of
        (dataset, fold, run_id, family) tuples that have cache files
        on disk.

    Returns
    -------
    list[dict]
        A new list (shallow copy of each dict) with `sweep_uncertain`
        added.  The original list is not mutated.
    """
    # Build a set of (dataset, fold) pairs that have any active GA cell
    uncertain_cells: set[tuple[str, int]] = {
        (ds, fold) for ds, fold, _run_id, _family in active_ga_cells
    }

    out: list[dict[str, Any]] = []
    for cfg in active_configs:
        annotated = dict(cfg)
        ds = cfg.get("dataset", "")
        fold = cfg.get("fold", -1)
        annotated["sweep_uncertain"] = (ds, fold) in uncertain_cells
        out.append(annotated)
    return out
