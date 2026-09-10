import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
pytest.importorskip("sklearn")

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing import projections as pj
from experiment_microscope.processing._binding import is_available
from experiment_microscope.views.latent_explorer import LatentExplorer


def test_pca_shapes_and_variance():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((30, 8))
    coords, evr = pj.pca(m, n_components=2)
    assert coords.shape == (30, 2)
    assert evr.size >= 1 and 0.0 <= evr[0] <= 1.0


def test_pca_pads_when_rank_deficient():
    m = np.ones((5, 4))  # rank 0 after centering
    coords, _ = pj.pca(m, n_components=3)
    assert coords.shape == (5, 3)


def test_tsne_rejects_tiny_input():
    with pytest.raises(ValueError):
        pj.tsne(np.zeros((3, 4)))


def test_view_declines_non_fold(qapp):
    v = LatentExplorer(DataRepository())
    v.show_node(TreeNode("x", "root", {"level": "root"}), "meeting01")
    assert "fold" in v._status.text()


@pytest.mark.skipif(not is_available(), reason="nn_microscope not built")
def test_real_latent_batch_and_project(qapp, first_fsdd_window):
    node, adapter = first_fsdd_window()
    if node is None or not adapter._snn_model_specs():
        pytest.skip("no FSDD corpus or trained model")
    batch = adapter.latent_batch("fsdd", 0, split="test", limit=24)
    assert batch["latents"].ndim == 2 and batch["latents"].shape[0] >= 1
    v = LatentExplorer(DataRepository())
    v.show_node(TreeNode("fold", "f", {"level": "fold", "dataset": "fsdd", "cv_fold": 0}),
                "meeting01")
    v._on_batch(batch)
    assert v._coords is not None and v._coords.shape[1] == 2

    seen = []
    v.sample_activated.connect(lambda *a: seen.append(a))
    # pyqtgraph ≥0.13 delivers `points` as a numpy array + a trailing event arg;
    # the handler must not do `not points` on it and must tolerate the extra arg.
    import numpy as np
    spot = type("P", (), {"data": lambda self: 0})()
    v._on_point_clicked(None, np.array([spot, spot], dtype=object), object())
    assert seen and seen[0][0] == "fsdd"
    seen.clear()
    v._on_point_clicked(None, np.array([], dtype=object))   # empty → no crash, no emit
    assert not seen
