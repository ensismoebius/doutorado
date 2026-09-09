import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.data.adapters import ParaconsistentPoint
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.paraconsistent_landscape import ParaconsistentLandscape, _facets


def _pt(run_tag, dt, dp, modality="eeg"):
    v = lambda x: Value(x, Origin.MEASURED)
    return ParaconsistentPoint(
        label=f"{run_tag} / fs", alpha=v(0.8), beta=v(0.1), g1=v(0.7), g2=v(-0.1),
        d_truth=v(dt), d_penalized=v(dp),
        facet={"run_tag": run_tag, "modality": modality, "strategy": "handcrafted"},
    )


class _Repo(DataRepository):
    def __init__(self, pts):
        super().__init__()
        self._pts = pts

    def all_paraconsistent_points(self):
        return self._pts


def test_facets_parsed_from_run_tag():
    fc = _facets(_pt("e05_p00_hc_daub10_lfcc_c1_eeg_rep0", 1.0, 1.2))
    assert fc["wavelet"] == "daub10" and fc["scale"] == "lfcc" and fc["modality"] == "eeg"


def test_filters_populate_and_narrow(qapp):
    v = ParaconsistentLandscape(_Repo([
        _pt("hc_daub10_lfcc_c1_eeg", 1.0, 1.2, "eeg"),
        _pt("hc_haar_mel_c1_voice", 0.9, 1.0, "voice"),
    ]))
    wav = v._combos["wavelet"]
    assert {wav.itemText(i) for i in range(wav.count())} == {"(all)", "daub10", "haar"}
    v._combos["modality"].setCurrentText("voice")
    assert len(v._selected()) == 1
    assert v._selected()[0].facet["run_tag"] == "hc_haar_mel_c1_voice"


def test_click_emits_point(qapp):
    v = ParaconsistentLandscape(_Repo([_pt("hc_haar_lfcc_c1_eeg", 1.0, 1.4)]))
    seen = []
    v.point_clicked.connect(seen.append)
    spot = v._scatter.points()[0]
    v._on_click(v._scatter, [spot])
    assert seen and seen[0].d_penalized.magnitude == 1.4
