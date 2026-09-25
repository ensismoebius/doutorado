#!/usr/bin/env python3
"""Tests for ga_cache.py — GaCacheTailer, pareto_frontier, GaSearchState."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "meeting01"))

from ga_cache import (
    GaCacheTailer,
    GaSearchState,
    _parse_identity,
    pareto_frontier,
)


# ── Filename regex ───────────────────────────────────────────────────────────

class TestParseIdentity:
    def test_snn(self):
        path = "/tmp/results/meeting01_ga_meeting01_loso_fsdd_fold0_run1_cache.jsonl"
        assert _parse_identity(path) == {
            "dataset": "fsdd", "fold": 0, "run_id": 1, "family": "snn",
        }

    def test_lstm(self):
        path = "/tmp/results/meeting01_ga_meeting01_loso_fsdd_fold0_run1_lstm_cache.jsonl"
        assert _parse_identity(path) == {
            "dataset": "fsdd", "fold": 0, "run_id": 1, "family": "lstm",
        }

    def test_gru(self):
        path = "/tmp/results/meeting01_ga_meeting01_loso_audiomnist_fold5_run3_gru_cache.jsonl"
        assert _parse_identity(path) == {
            "dataset": "audiomnist", "fold": 5, "run_id": 3, "family": "gru",
        }

    def test_transformer(self):
        path = "/tmp/results/meeting01_ga_meeting01_loso_eegmmidb_fold2_run10_transformer_cache.jsonl"
        assert _parse_identity(path) == {
            "dataset": "eegmmidb", "fold": 2, "run_id": 10, "family": "transformer",
        }

    def test_non_matching(self):
        assert _parse_identity("/tmp/results/meeting01_loso_fsdd_fold0_events.jsonl") is None

    def test_multidigit_fold_and_run(self):
        path = "/tmp/results/meeting01_ga_meeting01_loso_siena_fold12_run99_cache.jsonl"
        assert _parse_identity(path) == {
            "dataset": "siena", "fold": 12, "run_id": 99, "family": "snn",
        }


# ── GaCacheTailer ────────────────────────────────────────────────────────────

def _write_cache(path: str, individuals: list[dict]) -> None:
    with open(path, "w") as f:
        for ind in individuals:
            f.write(json.dumps(ind) + "\n")


class TestGaCacheTailer:
    def test_incremental_read(self, tmp_path):
        tag = "meeting01_loso"
        cache_file = tmp_path / f"meeting01_ga_{tag}_fsdd_fold0_run1_cache.jsonl"
        ind1 = {"genome": {}, "val_mse": 0.5, "param_count": 100,
                "inference_cost": 50, "feasible": True,
                "constraint_violation": 0.0, "objectives": [0.5, 50],
                "born_generation": 0}
        ind2 = {"genome": {}, "val_mse": 0.3, "param_count": 120,
                "inference_cost": 60, "feasible": True,
                "constraint_violation": 0.0, "objectives": [0.3, 60],
                "born_generation": 1}

        tailer = GaCacheTailer(str(tmp_path), tag)

        # First poll: nothing
        assert tailer.poll() == []

        # Write first individual
        _write_cache(str(cache_file), [ind1])
        result = tailer.poll()
        assert len(result) == 1
        assert result[0]["val_mse"] == 0.5
        assert result[0]["_dataset"] == "fsdd"
        assert result[0]["_family"] == "snn"

        # Second poll: nothing new
        assert tailer.poll() == []

        # Append second individual
        _write_cache(str(cache_file), [ind1, ind2])
        result = tailer.poll()
        assert len(result) == 1
        assert result[0]["val_mse"] == 0.3
        assert result[0]["_family"] == "snn"

    def test_truncated_line(self, tmp_path):
        tag = "meeting01_loso"
        cache_file = tmp_path / f"meeting01_ga_{tag}_fsdd_fold0_run1_lstm_cache.jsonl"
        ind = {"genome": {}, "val_mse": 0.1, "param_count": 50,
               "inference_cost": 30, "feasible": True,
               "constraint_violation": 0.0, "objectives": [0.1, 30],
               "born_generation": 0}

        tailer = GaCacheTailer(str(tmp_path), tag)

        # Write a complete line + a partial line (cut mid-JSON)
        with open(str(cache_file), "w") as f:
            f.write(json.dumps(ind) + "\n")
            f.write('{"genome": {}, "val_mse": 0.2, "param_count')
        result = tailer.poll()
        assert len(result) == 1  # only the complete line

        # Append the rest of the truncated line to complete the JSON
        with open(str(cache_file), "a") as f:
            f.write('": 60, "inference_cost": 40, "feasible": true, '
                    '"constraint_violation": 0.0, "objectives": [0.2, 40], '
                    '"born_generation": 1}\n')
        result = tailer.poll()
        assert len(result) == 1
        assert result[0]["val_mse"] == 0.2
        assert result[0]["_family"] == "lstm"

    def test_vanished_file(self, tmp_path):
        tag = "meeting01_loso"
        cache_file = tmp_path / f"meeting01_ga_{tag}_fsdd_fold0_run1_cache.jsonl"
        ind = {"genome": {}, "val_mse": 0.5, "param_count": 100,
               "inference_cost": 50, "feasible": True,
               "constraint_violation": 0.0, "objectives": [0.5, 50],
               "born_generation": 0}

        tailer = GaCacheTailer(str(tmp_path), tag)
        _write_cache(str(cache_file), [ind])
        tailer.poll()
        assert str(cache_file) in tailer.mtimes

        # Remove the file
        os.remove(str(cache_file))
        tailer.poll()
        assert str(cache_file) in tailer.missing
        assert str(cache_file) not in tailer.mtimes

    def test_active_cells(self, tmp_path):
        tag = "meeting01_loso"
        for fam in ["", "_lstm", "_gru"]:
            _write_cache(
                str(tmp_path / f"meeting01_ga_{tag}_fsdd_fold0_run1{fam}_cache.jsonl"),
                [{"genome": {}, "val_mse": 0.1, "param_count": 10,
                  "inference_cost": 5, "feasible": True,
                  "constraint_violation": 0.0, "objectives": [0.1, 5],
                  "born_generation": 0}],
            )
        tailer = GaCacheTailer(str(tmp_path), tag)
        cells = tailer.active_cells()
        assert len(cells) == 3
        families = {c[3] for c in cells}
        assert families == {"snn", "lstm", "gru"}


# ── pareto_frontier ──────────────────────────────────────────────────────────

class TestParetoFrontier:
    def test_basic(self):
        """Two non-dominated points should both be in the front."""
        inds = [
            {"val_mse": 0.1, "inference_cost": 100, "feasible": True},
            {"val_mse": 0.3, "inference_cost": 50, "feasible": True},
        ]
        front = pareto_frontier(inds)
        assert len(front) == 2

    def test_dominated(self):
        """A dominated point should be excluded."""
        inds = [
            {"val_mse": 0.1, "inference_cost": 100, "feasible": True},
            {"val_mse": 0.5, "inference_cost": 200, "feasible": True},  # dominated
        ]
        front = pareto_frontier(inds)
        assert len(front) == 1
        assert front[0]["val_mse"] == 0.1

    def test_tie(self):
        """Two identical points should both be in the front (neither dominates)."""
        inds = [
            {"val_mse": 0.1, "inference_cost": 100, "feasible": True},
            {"val_mse": 0.1, "inference_cost": 100, "feasible": True},
        ]
        front = pareto_frontier(inds)
        assert len(front) == 2

    def test_infeasible_excluded(self):
        """Infeasible points are excluded when feasible ones exist."""
        inds = [
            {"val_mse": 0.1, "inference_cost": 100, "feasible": True},
            {"val_mse": 0.05, "inference_cost": 80, "feasible": False,
             "constraint_violation": 1.0},
        ]
        front = pareto_frontier(inds)
        assert len(front) == 1
        assert front[0]["feasible"] is True

    def test_all_infeasible(self):
        """Among infeasible points, least violation is Pareto-optimal."""
        inds = [
            {"val_mse": 0.1, "inference_cost": 100, "feasible": False,
             "constraint_violation": 2.0},
            {"val_mse": 0.05, "inference_cost": 80, "feasible": False,
             "constraint_violation": 0.5},
        ]
        front = pareto_frontier(inds)
        assert len(front) == 1
        assert front[0]["constraint_violation"] == 0.5

    def test_none_objectives_skipped(self):
        """Individuals with None objectives are skipped in feasible front."""
        inds = [
            {"val_mse": 0.1, "inference_cost": 100, "feasible": True},
            {"val_mse": None, "inference_cost": 50, "feasible": True},
        ]
        front = pareto_frontier(inds)
        assert len(front) == 1

    def test_three_points_l_shape(self):
        """Classic L-shape: all three are non-dominated."""
        inds = [
            {"val_mse": 0.1, "inference_cost": 200, "feasible": True},
            {"val_mse": 0.3, "inference_cost": 100, "feasible": True},
            {"val_mse": 0.5, "inference_cost": 50, "feasible": True},
        ]
        front = pareto_frontier(inds)
        assert len(front) == 3


# ── GaSearchState ────────────────────────────────────────────────────────────

class TestGaSearchState:
    def test_summary_shape(self):
        state = GaSearchState("fsdd", 0, 1, "snn")
        state.add({"genome": {}, "val_mse": 0.5, "param_count": 100,
                    "inference_cost": 50, "feasible": True,
                    "constraint_violation": 0.0, "objectives": [0.5, 50],
                    "born_generation": 0})
        s = state.summary()
        assert s["dataset"] == "fsdd"
        assert s["fold"] == 0
        assert s["run_id"] == 1
        assert s["family"] == "snn"
        assert s["n_individuals"] == 1
        assert s["generations"] == [0]
        assert s["best_val_mse"] == 0.5
        assert s["mutation_count"] is None
        assert s["crossover_count"] is None
        assert "not recorded" in s["mutation_crossover_note"]

    def test_pareto_cached(self):
        state = GaSearchState("fsdd", 0, 1, "lstm")
        state.add({"genome": {}, "val_mse": 0.5, "param_count": 100,
                    "inference_cost": 50, "feasible": True,
                    "constraint_violation": 0.0, "objectives": [0.5, 50],
                    "born_generation": 0})
        p1 = state.pareto
        p2 = state.pareto
        assert p1 is p2  # cached, not recomputed

    def test_nan_val_mse(self):
        """NaN val_mse should serialize to null in summary."""
        state = GaSearchState("fsdd", 0, 1, "snn")
        state.add({"genome": {}, "val_mse": float("nan"), "param_count": 100,
                    "inference_cost": 50, "feasible": True,
                    "constraint_violation": 0.0, "objectives": [float("nan"), 50],
                    "born_generation": 0})
        s = state.summary()
        assert s["best_val_mse"] is None
