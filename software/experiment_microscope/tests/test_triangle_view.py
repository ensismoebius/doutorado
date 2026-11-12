import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.core.integrity import Origin
from experiment_microscope.data.adapters import FeatureMatrix, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing.thesis import HandcraftedSpec
from experiment_microscope.views.triangle_view import TriangleView

SIX = ("energy", "zcr", "entropy", "teager", "jitter", "shimmer")


def _spec(*, cepstral):
    # Every field declared, as a phase00 profile declares it.
    return HandcraftedSpec(transform="dtwpt", wavelet="haar", scale="lfcc", cepstral=cepstral,
                           modality="eeg", dtwpt_level=4, descriptors=SIX, fusion_mode=None)


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

    def handcrafted_spec(self, phase, tag):
        return _spec(cepstral=False)


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


def test_feature_band_map_non_cepstral(qapp):
    v = TriangleView(_repo())
    spec = _spec(cepstral=False)  # 6 descriptors
    v._map_features_to_bands(spec, n_energy_bands=16, n_features=96)
    assert (v._n_bands, v._per_band, v._cepstral_offset) == (16, 6, 0)
    assert v._band_for_feature(13) == 2
    assert v._feature_span_for_band(2) == (12, 18)


def test_feature_band_map_cepstral_offset(qapp):
    v = TriangleView(_repo())
    spec = _spec(cepstral=True)  # energy dropped -> 5 per band + 16 cepstral
    v._map_features_to_bands(spec, n_energy_bands=16, n_features=16 + 5 * 16)
    assert (v._n_bands, v._per_band, v._cepstral_offset) == (16, 5, 16)
    assert v._band_for_feature(10) is None          # inside the global cepstral coeffs
    assert v._band_for_feature(16) == 0


def test_feature_band_map_disabled_when_layout_not_one_to_one(qapp):
    v = TriangleView(_repo())
    v._map_features_to_bands(_spec(cepstral=False), n_energy_bands=16, n_features=50)
    assert v._n_bands == 0


def test_highlight_items_added_and_cleared(qapp):
    v = TriangleView(_repo())
    v._pw = v._layout.addPlot()
    v._pf = v._layout.addPlot()
    v._matrix = _FakeThesis()._fm
    v._run_node = TreeNode("run", "r", {"level": "run", "phase": "p", "run_tag": "hc_haar_lfcc_c1_eeg"})
    v._map_features_to_bands(_spec(cepstral=False), 4, 24)  # 4 bands x 6
    v._hl_band = 1
    v._apply_highlight()
    assert v._hl_line is not None and v._hl_region is not None
    v._hl_band = None
    v._apply_highlight()
    assert v._hl_line is None and v._hl_region is None


# -- HandcraftedSpec has no defaults ------------------------------------
def _config(**over):
    hc = {"transform": "dtwpt", "wavelet": "daub10", "scale": "lfcc", "cepstral": False,
          "dtwpt_level": 4, "descriptors": list(SIX)}
    cfg = {"dataset": {"modality": "eeg", "max_samples": 0},
           "feature_extraction": {"strategy": "handcrafted", "handcrafted": hc}}
    for k, v in over.items():
        if k in hc:
            hc[k] = v
        else:
            cfg["dataset"][k] = v
    return cfg


def test_spec_from_config_reads_every_declared_key():
    spec = HandcraftedSpec.from_config(_config())
    assert spec == HandcraftedSpec(transform="dtwpt", wavelet="daub10", scale="lfcc",
                                   cepstral=False, modality="eeg", dtwpt_level=4,
                                   descriptors=SIX, fusion_mode=None)


def test_spec_refuses_an_undeclared_key():
    cfg = _config()
    del cfg["feature_extraction"]["handcrafted"]["descriptors"]
    with pytest.raises(KeyError):
        HandcraftedSpec.from_config(cfg)


def test_spec_requires_fusion_mode_exactly_for_fused():
    with pytest.raises(KeyError):
        HandcraftedSpec.from_config(_config(modality="fused"))
    assert HandcraftedSpec.from_config(
        _config(modality="fused", fusion_mode="late")).fusion_mode == "late"
    with pytest.raises(ValueError):
        HandcraftedSpec(transform="dtwpt", wavelet="haar", scale="lfcc", cepstral=False,
                        modality="eeg", dtwpt_level=4, descriptors=SIX, fusion_mode="late")


def test_spec_refuses_a_non_handcrafted_run():
    cfg = _config()
    cfg["feature_extraction"] = {"strategy": "pca", "pca": {"n_components": 8}}
    with pytest.raises(ValueError):
        HandcraftedSpec.from_config(cfg)
