#!/usr/bin/env python3
"""Tests for sweep_state.py — annotate_sweep_uncertain."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "meeting01", "dashboard"))

from sweep_state import annotate_sweep_uncertain


class TestAnnotateSweepUncertain:
    def test_no_ga_cells(self):
        configs = [
            {"config_id": "c1", "dataset": "fsdd", "fold": 0},
            {"config_id": "c2", "dataset": "fsdd", "fold": 1},
        ]
        result = annotate_sweep_uncertain(configs, set())
        assert len(result) == 2
        assert all(c["sweep_uncertain"] is False for c in result)

    def test_matching_cell(self):
        configs = [
            {"config_id": "c1", "dataset": "fsdd", "fold": 0},
        ]
        cells = {("fsdd", 0, 1, "lstm")}
        result = annotate_sweep_uncertain(configs, cells)
        assert result[0]["sweep_uncertain"] is True

    def test_non_matching_cell(self):
        configs = [
            {"config_id": "c1", "dataset": "fsdd", "fold": 0},
        ]
        cells = {("fsdd", 1, 1, "lstm")}  # different fold
        result = annotate_sweep_uncertain(configs, cells)
        assert result[0]["sweep_uncertain"] is False

    def test_multiple_cells(self):
        configs = [
            {"config_id": "c1", "dataset": "fsdd", "fold": 0},
            {"config_id": "c2", "dataset": "audiomnist", "fold": 0},
            {"config_id": "c3", "dataset": "fsdd", "fold": 1},
        ]
        cells = {
            ("fsdd", 0, 1, "snn"),
            ("audiomnist", 0, 1, "lstm"),
        }
        result = annotate_sweep_uncertain(configs, cells)
        assert result[0]["sweep_uncertain"] is True
        assert result[1]["sweep_uncertain"] is True
        assert result[2]["sweep_uncertain"] is False

    def test_does_not_mutate_input(self):
        configs = [{"config_id": "c1", "dataset": "fsdd", "fold": 0}]
        cells = {("fsdd", 0, 1, "snn")}
        result = annotate_sweep_uncertain(configs, cells)
        assert configs[0].get("sweep_uncertain") is None  # original not mutated
        assert result[0]["sweep_uncertain"] is True

    def test_empty_configs(self):
        result = annotate_sweep_uncertain([], {("fsdd", 0, 1, "snn")})
        assert result == []

    def test_preserves_other_fields(self):
        configs = [{"config_id": "c1", "dataset": "fsdd", "fold": 0, "status": "running"}]
        result = annotate_sweep_uncertain(configs, set())
        assert result[0]["status"] == "running"
        assert result[0]["config_id"] == "c1"
