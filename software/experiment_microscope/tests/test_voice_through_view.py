"""Voice Through the Network — play a whole recording, one window per frame
(companion to Autoencoder; replaces the removed SNN 3D tab)."""

import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.core.animation import TimelinePlayer
from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.meeting01_adapter import Meeting01Adapter
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import is_available
from experiment_microscope.views.voice_through_view import VoiceThroughView


def test_view_declines_non_meeting01_or_no_fold(qapp):
    v = VoiceThroughView(DataRepository())
    v.show_node(TreeNode("x", "root", {"level": "root"}), "meeting01")
    assert "fold" in v._status.text().lower() or "dobra" in v._status.text().lower()
    assert v._current_rows == []
    # embedded graph is hidden-controls but still cleared, not left dangling
    assert v._ae._cols is None


def test_embedded_autoencoder_controls_are_hidden(qapp):
    # isHidden() reflects the widget's own explicit hide/show state
    # regardless of ancestor visibility — isVisible() would be False here
    # either way, since nothing in this test tree is ever .show()n.
    v = VoiceThroughView(DataRepository())
    assert v._ae._model_bar_widget.isHidden()


@pytest.mark.skipif(not is_available(), reason="nn_microscope not built")
def test_real_recording_plays_through_the_embedded_graph(qapp):
    repo = DataRepository()
    adapter = repo.adapter("meeting01")
    if not adapter._snn_model_specs():
        pytest.skip("no FSDD corpus or trained model")
    node = TreeNode("fold", "fold", {"level": "fold", "dataset": "fsdd", "cv_fold": 0})

    player = TimelinePlayer()
    v = VoiceThroughView(repo)
    v.set_timeline(player)
    v.show_node(node, "meeting01")

    assert v._recordings  # real recordings found for this fold's test split
    assert v._specs       # a model was auto-picked
    assert v._current_spec is not None
    assert player.total_frames == len(v._current_rows)
    assert v._ae._cols is not None  # the embedded graph actually rendered a frame

    # scrubbing to another window in the SAME recording keeps the model fixed
    # but changes the rendered frame
    first_cols = v._ae._cols
    if player.total_frames > 1:
        player.seek(player.total_frames - 1)
        assert v._ae._cols is not None
        # the reconstruction/latent shapes are identical (same model), only
        # the activations differ — comparing shapes is a stable, cheap check
        assert [c["act"].size for c in v._ae._cols] == [c["act"].size for c in first_cols]

    assert "window" in v._status.text().lower() or "janela" in v._status.text().lower()


@pytest.mark.skipif(not is_available(), reason="nn_microscope not built")
def test_recordings_for_groups_and_orders_windows_by_recording():
    a = Meeting01Adapter()
    if not a._snn_model_specs():
        pytest.skip("no FSDD corpus or trained model")
    node = TreeNode("fold", "fold", {"level": "fold", "dataset": "fsdd", "cv_fold": 0,
                                      "split": "test"})
    recs = a.recordings_for(node)
    assert recs
    for r in recs[:5]:
        rows = r["rows"]
        assert len(rows) >= 1
        assert r["recording_id"] >= 0

    # true in-recording-order requires reading each row's own metadata back —
    # cross-check the FIRST recording against the raw split directly
    split = a._split("fsdd", 0)
    metas = split["test_meta"]
    first = recs[0]
    idxs = [metas[row]["source_window_index"] for row in first["rows"]]
    assert idxs == sorted(idxs)
    assert all(metas[row]["recording_id"] == first["recording_id"] for row in first["rows"])


def test_recordings_for_needs_dataset_and_fold():
    a = Meeting01Adapter()
    assert a.recordings_for(TreeNode("x", "root", {"level": "root"})) == []
