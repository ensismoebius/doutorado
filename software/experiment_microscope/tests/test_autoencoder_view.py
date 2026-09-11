"""The unified encoder+decoder graph view (FIXME §18, §19)."""

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.core.animation import TimelinePlayer
from experiment_microscope.processing._binding import is_available
from experiment_microscope.views.autoencoder_view import AutoencoderView


class _Spot:
    def __init__(self, d):
        self._d = d

    def data(self):
        return self._d


def test_view_declines_non_window(qapp):
    v = AutoencoderView(DataRepository())
    v.show_node(TreeNode("x", "root", {"level": "root"}), "meeting01")
    assert "window" in v._readout.text().lower() or "janela" in v._readout.text().lower()
    assert v._cols is None


@pytest.mark.skipif(not is_available(), reason="nn_microscope not built")
def test_full_graph_builds_and_animates(qapp):
    repo = DataRepository()
    adapter = repo.adapter("meeting01")
    specs = adapter._snn_model_specs()
    if not specs:
        pytest.skip("no trained SNN-AE .npz")
    s = specs[0]
    node = TreeNode(
        kind="sample", label="w",
        handle={"level": "window", "dataset": s["dataset"], "cv_fold": s["fold"],
                "split": "test", "row": 0, "encoding": s["encoding"],
                "architecture": s["architecture"]},
    )
    player = TimelinePlayer()
    v = AutoencoderView(repo)
    v.set_timeline(player)
    v.show_node(node, "meeting01")

    assert v._cols is not None
    # input → … → latent → … → reconstruction, both halves present
    halves = {c["half"] for c in v._cols}
    assert {"in", "enc", "dec"} <= halves
    assert v._cols[-1]["kind"] == "recon"
    assert any(c["kind"] == "latent" for c in v._cols)
    # one animation frame per column
    assert player.total_frames == len(v._cols)

    # a LIF column carries a real membrane snapshot + threshold
    lif = [c for c in v._cols if c["vmem"] is not None]
    assert lif and lif[0]["vth"] is not None

    # flood animation just re-renders, no raise
    v._on_frame(0)
    v._on_frame(len(v._cols) - 1)

    # click a LIF neuron → readout names membrane vs firing line
    li = v._cols.index(lif[0])
    v._on_click(None, np.array([_Spot((li, 3))], dtype=object), object())
    txt = v._readout.text().lower()
    assert "membrane" in txt or "carga" in txt

    # empty click payload is harmless
    v._on_click(None, np.array([], dtype=object))

    # detail mode toggles without raising
    v._detail_cb.setChecked(True)
    v._detail_cb.setChecked(False)

    # the model picker lists at least the auto-picked spec, and re-running the
    # explicitly chosen model re-emits trace_changed (Model Structure follows it)
    assert v._model_combo.count() >= 1
    seen = []
    v.trace_changed.connect(lambda tr, sp: seen.append(sp))
    v._model_combo.setCurrentIndex(0)
    v._run_selected_model()
    assert seen and seen[-1]["dataset"] == s["dataset"]
    assert "explicitly selected" in v._readout.text() or "selecionado manualmente" in v._readout.text()


def test_run_selected_model_needs_a_pick(qapp):
    v = AutoencoderView(DataRepository())
    v._node = None
    v._run_selected_model()  # no node selected yet — must not raise
    assert v._cols is None


def test_compare_all_needs_a_node(qapp):
    v = AutoencoderView(DataRepository())
    v._node = None
    v._compare_all()  # no node selected yet — must not raise
    assert v._compare_rows == []


@pytest.mark.skipif(not is_available(), reason="nn_microscope not built")
def test_compare_all_fills_table_and_row_click_loads_model(qapp):
    repo = DataRepository()
    adapter = repo.adapter("meeting01")
    specs = adapter._snn_model_specs()
    if not specs:
        pytest.skip("no trained SNN-AE .npz")
    s = specs[0]
    node = TreeNode(
        kind="sample", label="w",
        handle={"level": "window", "dataset": s["dataset"], "cv_fold": s["fold"],
                "split": "test", "row": 0},
    )
    v = AutoencoderView(repo)
    v.show_node(node, "meeting01")
    v._compare_all()
    assert v._compare_rows  # at least this fold's own models were compared
    assert v._compare_table.isVisible()
    assert v._compare_table.rowCount() == len(v._compare_rows)
    # every compared row belongs to THIS window's exact fold — never a
    # cross-fold substitute (FIXME §33 no-substitution rule)
    assert all(r["spec"]["fold"] == s["fold"] for r in v._compare_rows)

    seen = []
    v.trace_changed.connect(lambda tr, sp: seen.append(sp))
    v._on_compare_row(0, 0)
    assert seen and seen[-1]["fold"] == s["fold"]
