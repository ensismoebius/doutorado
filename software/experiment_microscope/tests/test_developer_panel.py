"""Developer panel (FIXME §37): cache stats + shape consistency verdict."""

from __future__ import annotations

from experiment_microscope.core.cache import CacheKey, TransformationCache
from experiment_microscope.views.developer_panel import DeveloperPanel


def test_cache_tracks_hit_rate():
    cache = TransformationCache()
    key = CacheKey.make("a", "s", "stage", {"p": 1})
    assert cache.stats()["hit_rate"] == 0.0
    cache.compute_blocking(key, lambda: 42)   # miss
    cache.compute_blocking(key, lambda: 99)   # hit -> still 42
    st = cache.stats()
    assert st["hits"] == 1 and st["misses"] == 1
    assert st["hit_rate"] == 0.5
    assert st["entries"] == 1


def test_panel_flags_non_power_of_two_transform(qapp):
    cache = TransformationCache()
    shapes = {"raw": (256,), "transformed": (300,), "latent": None, "reconstruction": None}
    panel = DeveloperPanel(cache, lambda: shapes)
    panel.refresh()
    assert "LOUD" in panel._l_verdict.text()
    assert "power of two" in panel._l_verdict.text()


def test_panel_flags_reconstruction_length_mismatch(qapp):
    cache = TransformationCache()
    shapes = {"raw": (256,), "transformed": (256,), "latent": (8,), "reconstruction": (255,)}
    panel = DeveloperPanel(cache, lambda: shapes)
    panel.refresh()
    assert "LOUD" in panel._l_verdict.text()


def test_panel_consistent_and_missing_render_as_dash(qapp):
    cache = TransformationCache()
    shapes = {"raw": (256,), "transformed": (256,), "latent": None, "reconstruction": (256,)}
    panel = DeveloperPanel(cache, lambda: shapes)
    panel.refresh()
    assert panel._l_verdict.text() == "consistent"
    assert panel._l_stage["latent"].text() == "—"


def test_panel_no_selection_verdict(qapp):
    panel = DeveloperPanel(TransformationCache(), lambda: {})
    panel.refresh()
    assert "no live representation" in panel._l_verdict.text()
