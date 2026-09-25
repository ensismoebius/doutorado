#!/usr/bin/env python3
"""Tests for monitor.py's JSON accessor methods."""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "meeting01"))

from monitor import (
    ConfigState,
    FoldCell,
    ProcState,
    SessionState,
    _config_state_to_dict,
    _finite_or_none,
    _fold_cell_to_dict,
    _proc_state_to_dict,
)


# ── _finite_or_none ──────────────────────────────────────────────────────────

class TestFiniteOrNone:
    def test_normal_float(self):
        assert _finite_or_none(3.14) == 3.14

    def test_nan(self):
        assert _finite_or_none(float("nan")) is None

    def test_inf(self):
        assert _finite_or_none(float("inf")) is None

    def test_neg_inf(self):
        assert _finite_or_none(float("-inf")) is None

    def test_int(self):
        assert _finite_or_none(42) == 42

    def test_none(self):
        assert _finite_or_none(None) is None

    def test_string(self):
        assert _finite_or_none("hello") == "hello"

    def test_dict_nested(self):
        d = {"a": float("nan"), "b": {"c": float("inf"), "d": 1.0}}
        result = _finite_or_none(d)
        assert result == {"a": None, "b": {"c": None, "d": 1.0}}

    def test_list_nested(self):
        lst = [float("nan"), 1.0, [float("inf"), 2.0]]
        result = _finite_or_none(lst)
        assert result == [None, 1.0, [None, 2.0]]


# ── ConfigState dict ─────────────────────────────────────────────────────────

class TestConfigStateToDict:
    def test_basic(self):
        c = ConfigState(
            config_id="cfg_001", dataset="fsdd", fold=0,
            model="snn", encoding="direct", role="snn_final",
            run_id=1, seed=42, hyperparams={"v_th": 0.5},
            max_epochs=50, lr=0.001, param_count=1000, macs=5000,
            epochs=[(1, 0.5, 0.6), (2, 0.3, 0.4)],
            status="done", stop_reason="early_stop",
            epochs_run=2, best_val=0.4, best_epoch=2,
            metrics={"val": {"mse": 0.4}, "test": {"mse": 0.35}},
        )
        d = _config_state_to_dict(c)
        assert d["config_id"] == "cfg_001"
        assert d["dataset"] == "fsdd"
        assert d["fold"] == 0
        assert d["model"] == "snn"
        assert d["status"] == "done"
        assert d["epochs"] == [
            {"epoch": 1, "train": 0.5, "val": 0.6},
            {"epoch": 2, "train": 0.3, "val": 0.4},
        ]
        assert d["last_train"] == 0.3
        assert d["last_val"] == 0.4
        assert d["gap"] == pytest.approx(0.1)

    def test_nan_in_epochs(self):
        c = ConfigState(config_id="cfg_nan")
        c.epochs = [(1, float("nan"), 0.5)]
        d = _config_state_to_dict(c)
        assert d["epochs"][0]["train"] is None
        assert d["epochs"][0]["val"] == 0.5

    def test_json_roundtrip(self):
        """Ensure no NaN/Inf tokens in the JSON output."""
        c = ConfigState(config_id="cfg_rt")
        c.epochs = [(1, float("nan"), float("inf"))]
        d = _config_state_to_dict(c)
        s = json.dumps(d)
        assert "NaN" not in s
        assert "Infinity" not in s
        parsed = json.loads(s)
        assert parsed["epochs"][0]["train"] is None
        assert parsed["epochs"][0]["val"] is None


# ── ProcState dict ───────────────────────────────────────────────────────────

class TestProcStateToDict:
    def test_basic(self):
        p = ProcState(path="/tmp/events.jsonl", dataset="fsdd", fold=0,
                      started=True, ended=False, error="")
        d = _proc_state_to_dict(p)
        assert d["path"] == "/tmp/events.jsonl"
        assert d["started"] is True
        assert d["ended"] is False
        assert d["error"] == ""


# ── FoldCell dict ────────────────────────────────────────────────────────────

class TestFoldCellToDict:
    def test_running(self):
        cell = FoldCell(done=2, running=1, failed=0, total=5)
        d = _fold_cell_to_dict("fsdd", 0, cell)
        assert d["dataset"] == "fsdd"
        assert d["fold"] == 0
        assert d["done"] == 2
        assert d["running"] == 1
        assert d["frac"] == pytest.approx(0.4)
        assert d["status"] == "running"

    def test_done(self):
        cell = FoldCell(done=5, running=0, failed=0, total=5)
        d = _fold_cell_to_dict("fsdd", 0, cell)
        assert d["status"] == "done"
        assert d["frac"] == 1.0

    def test_failed(self):
        cell = FoldCell(done=3, running=0, failed=1, total=5)
        d = _fold_cell_to_dict("fsdd", 0, cell)
        assert d["status"] == "failed"


# ── SessionState JSON accessors ──────────────────────────────────────────────

class TestSessionStateJson:
    def _make_state(self) -> SessionState:
        ss = SessionState()
        ss.session = {
            "run_tag": "meeting01_loso",
            "cv_num_folds": 6,
            "repeats": 5,
            "search_space": {
                "baselines": ["lstm", "gru", "transformer"],
                "encodings": ["direct", "latency", "poisson"],
                "snn_architectures": ["spiking_autoencoder"],
                "ga_population_size": 8,
                "ga_generations": 5,
            },
        }
        # Add a running config
        c1 = ConfigState(config_id="cfg_1", dataset="fsdd", fold=0,
                         model="snn", encoding="direct", status="running")
        c1.epochs = [(1, 0.5, 0.6)]
        c1.last_ts = 100.0
        ss.configs["cfg_1"] = c1

        # Add a done config
        c2 = ConfigState(config_id="cfg_2", dataset="fsdd", fold=0,
                         model="lstm", encoding="direct", role="baseline",
                         status="done", best_val=0.3, best_epoch=5)
        c2.epochs = [(i, 0.5 - i * 0.05, 0.6 - i * 0.05) for i in range(1, 6)]
        c2.metrics = {"val": {"mse": 0.3}}
        ss.configs["cfg_2"] = c2

        # Add a proc
        ss.procs["/tmp/events.jsonl"] = ProcState(
            path="/tmp/events.jsonl", dataset="fsdd", fold=0,
            started=True, ended=False,
        )

        # Add an event
        ss.events.append({"type": "epoch", "v": 1, "config_id": "cfg_1",
                          "epoch": 1, "train_loss": 0.5, "val_loss": 0.6})
        return ss

    def test_session_summary_json(self):
        ss = self._make_state()
        s = ss.session_summary_json()
        assert "counts" in s
        assert "eta_seconds" in s
        assert "grid_size" in s

    def test_fold_grid_json(self):
        ss = self._make_state()
        fg = ss.fold_grid_json()
        assert isinstance(fg, list)
        assert len(fg) >= 1
        assert "dataset" in fg[0]
        assert "status" in fg[0]

    def test_active_configs_json(self):
        ss = self._make_state()
        ac = ss.active_configs_json()
        assert len(ac) == 1
        assert ac[0]["config_id"] == "cfg_1"

    def test_completed_configs_json(self):
        ss = self._make_state()
        cc = ss.completed_configs_json()
        assert len(cc) == 1
        assert cc[0]["config_id"] == "cfg_2"

    def test_events_json(self):
        ss = self._make_state()
        ev = ss.events_json()
        assert len(ev) == 1
        assert ev[0]["type"] == "epoch"

    def test_full_snapshot_json(self):
        ss = self._make_state()
        snap = ss.full_snapshot_json()
        assert "summary" in snap
        assert "fold_grid" in snap
        assert "active_configs" in snap
        assert "completed_configs" in snap
        assert "events" in snap
        assert "procs" in snap

    def test_full_snapshot_json_no_nan(self):
        """NaN loss must serialize to null, never the literal NaN token."""
        ss = SessionState()
        ss.session = {"run_tag": "test"}
        c = ConfigState(config_id="cfg_nan")
        c.epochs = [(1, float("nan"), float("inf"))]
        ss.configs["cfg_nan"] = c
        snap = ss.full_snapshot_json()
        s = json.dumps(snap)
        assert "NaN" not in s
        assert "Infinity" not in s

    def test_marginals_json(self):
        ss = self._make_state()
        m = ss.marginals_json()
        assert isinstance(m, dict)
        # marginals only populate for snn_sweep/snn_final roles
        # with hyperparams like architecture/v_th/alpha, so may be empty here
        assert "architecture" in m

    def test_aggregation_json(self):
        ss = self._make_state()
        ag = ss.aggregation_json()
        assert isinstance(ag, list)
        # Should have at least the done config's group
        assert len(ag) >= 1
        assert "model" in ag[0]
        assert "mean" in ag[0]
