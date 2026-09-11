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


class _FakeAdapter:
    """A fake meeting01 adapter with two folds: fold 0 has a trained model,
    fold 1 does not — exercises the multi-fold checklist and the
    skip-don't-substitute rule without needing a real build."""

    def list_folds(self, dataset):
        return [0, 1, 2]

    def _snn_model_specs(self):
        return [{"dataset": "fsdd", "fold": 0, "encoding": "direct",
                  "architecture": "dense", "encoder": "e.npz", "decoder": "d.npz"}]

    def latent_batch(self, dataset, cv_fold, *, split="test", limit=120):
        assert (dataset, cv_fold) == ("fsdd", 0)  # never called for fold 1
        n = 5
        return {
            "latents": np.tile(np.array([[float(cv_fold), 1.0]]), (n, 1)),
            "digits": list(range(n)), "speakers": ["s"] * n, "rows": list(range(n)),
            "split": split, "spec": {"architecture": "dense", "encoding": "direct"},
            "lif_params": True,
        }


def test_fold_checklist_populates_and_defaults_to_the_selected_fold(qapp):
    repo = DataRepository()
    repo._adapters["meeting01"] = _FakeAdapter()
    v = LatentExplorer(repo)
    v.show_node(TreeNode("fold", "f", {"level": "fold", "dataset": "fsdd", "cv_fold": 1}),
                "meeting01")
    assert v._folds.count() == 3
    assert v._checked_folds() == [1]

    # selecting a window in fold 2 of the SAME dataset adds it to the comparison
    v.show_node(TreeNode("w", "w", {"level": "window", "dataset": "fsdd", "cv_fold": 2}),
                "meeting01")
    assert sorted(v._checked_folds()) == [1, 2]


def test_multi_fold_batch_merges_and_skips_untrained_folds(qapp):
    repo = DataRepository()
    repo._adapters["meeting01"] = _FakeAdapter()
    v = LatentExplorer(repo)
    v.show_node(TreeNode("fold", "f", {"level": "fold", "dataset": "fsdd", "cv_fold": 0}),
                "meeting01")
    v._set_fold_checked(1, True)   # fold 1 has no trained model — must be skipped, not faked
    seen = {}
    from experiment_microscope.views.latent_explorer import _BatchWorker
    worker = _BatchWorker(repo.adapter("meeting01"), "fsdd", [0, 1], "test", 120)
    worker.done.connect(lambda b: seen.setdefault("batch", b))
    worker.failed.connect(lambda m: seen.setdefault("fail", m))
    worker.run()
    assert "fail" not in seen
    batch = seen["batch"]
    assert set(batch["fold"]) == {0}          # only the trained fold contributed
    assert batch["skipped"] and batch["skipped"][0][0] == 1
    v._on_batch(batch)
    assert "skipped 1 fold" in v._status.text()


@pytest.mark.skipif(not is_available(), reason="nn_microscope not built")
def test_real_latent_batch_and_project(qapp, first_fsdd_window):
    node, adapter = first_fsdd_window()
    if node is None or not adapter._snn_model_specs():
        pytest.skip("no FSDD corpus or trained model")
    batch = adapter.latent_batch("fsdd", 0, split="test", limit=24)
    assert batch["latents"].ndim == 2 and batch["latents"].shape[0] >= 1
    batch["fold"] = [0] * batch["latents"].shape[0]  # _BatchWorker tags every latent with its fold
    batch["skipped"] = []
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


def test_3d_panel_gets_a_visible_coordinate_system(qapp):
    """The 3-D PCA branch must draw a bounding box too, honestly labelled —
    PCA axes are directions, not physical measurements (FIXME §11, §19)."""
    pytest.importorskip("pyvista")
    rng = np.random.default_rng(0)
    n = 12
    batch = {
        "latents": rng.standard_normal((n, 6)),
        "digits": list(range(n)), "speakers": ["s"] * n, "rows": list(range(n)),
        "fold": [0] * n, "split": "test",
        "spec": {"architecture": "dense", "encoding": "direct"},
        "lif_params": True, "skipped": [],
    }
    v = LatentExplorer(DataRepository())
    v._ds = "fsdd"
    v._dims.setCurrentText("3-D (PCA)")
    v._on_batch(batch)
    if not (v._panel.available):
        pytest.skip("no usable render context")
    xmin, xmax, ymin, ymax, zmin, zmax = v._panel.plotter.bounds
    assert xmax > xmin or ymax > ymin or zmax > zmin  # something was actually drawn
