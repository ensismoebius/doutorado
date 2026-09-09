import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.core.integrity import Origin
from experiment_microscope.data.adapters import FeatureMatrix, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.triangle_view import TriangleView


class _FakeThesis:
    def __init__(self):
        self._fm = FeatureMatrix(
            values=np.random.default_rng(0).standard_normal((5, 8)),
            feature_names=tuple(f"f{j}" for j in range(8)),
            sample_labels=tuple(f"s{i}" for i in range(5)),
            class_labels=(1, 1, 2, 2, 3),
            origin=Origin.COMPUTED,
            set_label="fake / handcrafted-lfcc",
        )

    def load_features(self, node):
        return self._fm

    def paraconsistent_points(self):
        return []

    def _summary(self, phase, tag):
        return {"modality": "eeg", "handcrafted": {"wavelet": "haar", "dtwpt_level": 4}}


def _repo():
    r = DataRepository()
    r._adapters["thesis"] = _FakeThesis()
    return r


def test_triangle_ignores_non_run(qapp):
    v = TriangleView(_repo())
    v.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    assert v._run_node is None


def test_triangle_binds_run_and_sets_sample_range(qapp):
    v = TriangleView(_repo())
    node = TreeNode("run", "r", {"level": "run", "phase": "phase00", "run_tag": "t"})
    v.show_node(node, "thesis")
    assert v._run_node is node
    assert v._sample.maximum() == 4  # 5 samples
    v._sample.setValue(3)  # scrubbing must not raise
    assert "sample 3/4" in v._status.text()


# -- feature bar <-> wavelet band mapping (FIXME §44) --------------------
from experiment_microscope.processing.thesis import HandcraftedSpec  # noqa: E402


def test_feature_band_map_non_cepstral(qapp):
    v = TriangleView(_repo())
    spec = HandcraftedSpec(cepstral=False)  # 6 descriptors
    v._map_features_to_bands(spec, n_energy_bands=16, n_features=96)
    assert (v._n_bands, v._per_band, v._cepstral_offset) == (16, 6, 0)
    assert v._band_for_feature(13) == 2
    assert v._feature_span_for_band(2) == (12, 18)


def test_feature_band_map_cepstral_offset(qapp):
    v = TriangleView(_repo())
    spec = HandcraftedSpec(cepstral=True)  # energy dropped -> 5 per band + 16 cepstral
    v._map_features_to_bands(spec, n_energy_bands=16, n_features=16 + 5 * 16)
    assert (v._n_bands, v._per_band, v._cepstral_offset) == (16, 5, 16)
    assert v._band_for_feature(10) is None          # inside the global cepstral coeffs
    assert v._band_for_feature(16) == 0


def test_feature_band_map_disabled_when_layout_not_one_to_one(qapp):
    v = TriangleView(_repo())
    v._map_features_to_bands(HandcraftedSpec(), n_energy_bands=16, n_features=50)
    assert v._n_bands == 0


def test_highlight_items_added_and_cleared(qapp):
    v = TriangleView(_repo())
    v._pw = v._layout.addPlot()
    v._pf = v._layout.addPlot()
    v._matrix = _FakeThesis()._fm
    v._run_node = TreeNode("run", "r", {"level": "run", "phase": "p", "run_tag": "hc_haar_lfcc_c1_eeg"})
    v._map_features_to_bands(HandcraftedSpec(), 4, 24)  # 4 bands x 6
    v._hl_band = 1
    v._apply_highlight()
    assert v._hl_line is not None and v._hl_region is not None
    v._hl_band = None
    v._apply_highlight()
    assert v._hl_line is None and v._hl_region is None
