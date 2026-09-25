#!/usr/bin/env python3
"""downsample.py — stride and LTTB downsampling for large time series.

Provides two strategies for reducing the number of points in a series before
sending to the frontend:

  stride  — keep every Nth point (fast, preserves temporal alignment).
  LTTB    — Largest Triangle Three Buckets (preserves visual shape, slower).

Full resolution always stays on disk / in the backend's accumulator.  The
dashboard endpoint re-derives at higher resolution on a zoomed x-range on
demand.

LTTB reference: Hervás et al., "Real-Time Glyphified Visualization of
Large-Scale Simulation Data for Earth Sciences", 2012.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence


def stride_downsample(
    xs: Sequence[float],
    ys: Sequence[float],
    step: int,
) -> tuple[list[float], list[float]]:
    """Keep every *step*-th point.  Always includes the last point.

    Parameters
    ----------
    xs, ys : sequences of equal length
    step : int, must be >= 1

    Returns
    -------
    (xs_out, ys_out) — lists, possibly shorter than input.
    """
    if step <= 0:
        raise ValueError(f"step must be >= 1, got {step}")
    if not xs or len(xs) != len(ys):
        raise ValueError(f"xs and ys must have equal length (got {len(xs)} vs {len(ys)})")
    if step == 1 or len(xs) <= step:
        return list(xs), list(ys)
    out_x: list[float] = []
    out_y: list[float] = []
    for i in range(0, len(xs), step):
        out_x.append(xs[i])
        out_y.append(ys[i])
    # Ensure the last point is included
    if out_x[-1] != xs[-1]:
        out_x.append(xs[-1])
        out_y.append(ys[-1])
    return out_x, out_y


def lttb_downsample(
    xs: Sequence[float],
    ys: Sequence[float],
    target_count: int,
) -> tuple[list[float], list[float]]:
    """Largest Triangle Three Buckets downsampling.

    Reduces *xs*, *ys* to approximately *target_count* points while
    preserving the visual shape of the signal.  The first and last points
    are always kept.

    Parameters
    ----------
    xs, ys : sequences of equal length
    target_count : int, must be >= 3 (or == len(xs) to return unchanged)

    Returns
    -------
    (xs_out, ys_out)
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"xs and ys must have equal length (got {n} vs {len(ys)})")
    if target_count <= 0:
        raise ValueError(f"target_count must be >= 1, got {target_count}")
    if target_count >= n or n <= 2:
        return list(xs), list(ys)
    if target_count < 3:
        target_count = 3

    # Bucket size (excluding first and last points which are always kept)
    bucket_size = (n - 2) / (target_count - 2)

    out_x: list[float] = [xs[0]]
    out_y: list[float] = [ys[0]]

    a_index = 0  # previous selected point index

    for i in range(1, target_count - 1):
        # Compute the range of points in the current bucket
        bucket_start = int(math.floor((i - 1) * bucket_size)) + 1
        bucket_end = int(math.floor(i * bucket_size)) + 1
        bucket_end = min(bucket_end, n - 1)

        # Compute the range of points in the NEXT bucket (for averaging)
        next_start = int(math.floor(i * bucket_size)) + 1
        next_end = int(math.floor((i + 1) * bucket_size)) + 1
        next_end = min(next_end, n - 1)

        # Average of next bucket
        avg_x = 0.0
        avg_y = 0.0
        next_count = next_end - next_start
        if next_count > 0:
            for j in range(next_start, next_end):
                avg_x += xs[j]
                avg_y += ys[j]
            avg_x /= next_count
            avg_y /= next_count

        # Select the point in the current bucket with largest triangle area
        max_area = -1.0
        max_index = bucket_start
        for j in range(bucket_start, bucket_end):
            area = abs(
                (xs[a_index] - avg_x) * (ys[j] - ys[a_index])
                - (xs[a_index] - xs[j]) * (avg_y - ys[a_index])
            )
            if area > max_area:
                max_area = area
                max_index = j

        out_x.append(xs[max_index])
        out_y.append(ys[max_index])
        a_index = max_index

    out_x.append(xs[-1])
    out_y.append(ys[-1])
    return out_x, out_y


def auto_downsample(
    xs: Sequence[float],
    ys: Sequence[float],
    max_points: int = 500,
    method: str = "lttb",
) -> tuple[list[float], list[float]]:
    """Downsample if the series exceeds *max_points*, otherwise return as-is.

    Parameters
    ----------
    xs, ys : sequences of equal length
    max_points : maximum number of points to return
    method : "lttb" (default, visual fidelity) or "stride" (fast)
    """
    if len(xs) <= max_points:
        return list(xs), list(ys)
    if method == "stride":
        step = math.ceil(len(xs) / max_points)
        return stride_downsample(xs, ys, step)
    return lttb_downsample(xs, ys, max_points)
