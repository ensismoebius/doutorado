#!/usr/bin/env python3
"""Tests for downsample.py — stride and LTTB downsampling."""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "meeting01", "dashboard"))

from downsample import auto_downsample, lttb_downsample, stride_downsample


# ── stride_downsample ────────────────────────────────────────────────────────

class TestStrideDownsample:
    def test_step_1_returns_copy(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [10.0, 20.0, 30.0, 40.0]
        rx, ry = stride_downsample(xs, ys, 1)
        assert rx == xs
        assert ry == ys

    def test_step_2(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 20.0, 30.0, 40.0, 50.0]
        rx, ry = stride_downsample(xs, ys, 2)
        assert rx == [0.0, 2.0, 4.0]
        assert ry == [10.0, 30.0, 50.0]

    def test_last_point_always_included(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [0.0] * 6
        rx, ry = stride_downsample(xs, ys, 3)
        assert rx[-1] == 5.0

    def test_step_larger_than_data(self):
        xs = [0.0, 1.0]
        ys = [10.0, 20.0]
        rx, ry = stride_downsample(xs, ys, 10)
        assert rx == xs
        assert ry == ys

    def test_invalid_step(self):
        with pytest.raises(ValueError, match="step must be >= 1"):
            stride_downsample([0.0], [0.0], 0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            stride_downsample([0.0, 1.0], [0.0], 1)


# ── lttb_downsample ──────────────────────────────────────────────────────────

class TestLtTbDownsample:
    def test_no_reduction_needed(self):
        xs = [0.0, 1.0, 2.0]
        ys = [0.0, 1.0, 0.0]
        rx, ry = lttb_downsample(xs, ys, 10)
        assert rx == xs
        assert ry == ys

    def test_preserves_first_and_last(self):
        n = 100
        xs = list(range(n))
        ys = [math.sin(x * 0.1) for x in xs]
        rx, ry = lttb_downsample(xs, ys, 10)
        assert rx[0] == 0.0
        assert rx[-1] == 99.0

    def test_reduces_to_target(self):
        n = 200
        xs = list(range(n))
        ys = [float(x) for x in xs]
        rx, ry = lttb_downsample(xs, ys, 20)
        assert len(rx) == 20
        assert len(ry) == 20

    def test_target_3(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 1.0, 0.0, 1.0, 0.0]
        rx, ry = lttb_downsample(xs, ys, 3)
        assert len(rx) == 3
        assert rx[0] == 0.0
        assert rx[-1] == 4.0

    def test_monotonic_signal(self):
        """LTTB on a monotonic signal should keep endpoints and pick
        representative interior points."""
        n = 50
        xs = list(range(n))
        ys = [float(x) for x in xs]
        rx, ry = lttb_downsample(xs, ys, 5)
        assert len(rx) == 5
        assert rx[0] == 0.0
        assert rx[-1] == 49.0

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="target_count must be >= 1"):
            lttb_downsample([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            lttb_downsample([0.0, 1.0], [0.0], 2)


# ── auto_downsample ──────────────────────────────────────────────────────────

class TestAutoDownsample:
    def test_below_threshold(self):
        xs = list(range(10))
        ys = [float(x) for x in xs]
        rx, ry = auto_downsample(xs, ys, max_points=20)
        assert rx == xs

    def test_above_threshold_lttb(self):
        n = 500
        xs = list(range(n))
        ys = [float(x) for x in xs]
        rx, ry = auto_downsample(xs, ys, max_points=50, method="lttb")
        assert len(rx) == 50

    def test_above_threshold_stride(self):
        n = 500
        xs = list(range(n))
        ys = [float(x) for x in xs]
        rx, ry = auto_downsample(xs, ys, max_points=50, method="stride")
        assert len(rx) <= 51  # stride may include extra last point

    def test_exact_threshold(self):
        xs = list(range(50))
        ys = [float(x) for x in xs]
        rx, ry = auto_downsample(xs, ys, max_points=50)
        assert rx == xs
