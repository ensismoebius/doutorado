import pytest

pytest.importorskip("PySide6")

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.data.adapters import ParaconsistentPoint
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.ranking_view import _COLS, RankingView


def _pt(tag, dp, phase="phase00"):
    v = lambda x: Value(x, Origin.MEASURED)
    return ParaconsistentPoint(
        label=f"{tag} / fs", alpha=v(0.8), beta=v(0.1), g1=v(0.7), g2=v(-0.1),
        d_truth=v(dp - 0.1), d_penalized=v(dp),
        facet={"phase": phase, "run_tag": tag, "feature_set": "fs", "modality": "eeg", "seed": 42},
    )


class _Repo(DataRepository):
    def __init__(self, pts):
        super().__init__()
        self._pts = pts

    def all_paraconsistent_points(self):
        return self._pts


def test_sorted_by_d_penalized_ascending(qapp):
    v = RankingView(_Repo([_pt("b", 1.8), _pt("a", 0.3), _pt("c", 0.9)]))
    col = _COLS.index("d_penalized")
    vals = [v._table.item(r, col)._key for r in range(v._table.rowCount())]
    assert vals == sorted(vals)
    assert v._table.item(0, _COLS.index("run_tag")).text() == "a"


def test_filter_narrows_rows(qapp):
    v = RankingView(_Repo([_pt("alpha_run", 1.0), _pt("beta_run", 1.0)]))
    assert v._table.rowCount() == 2
    v._filter.setText("alpha")
    assert v._table.rowCount() == 1


def test_missing_metric_renders_dash_and_sorts_last(qapp):
    p = _pt("x", 1.0)
    p = ParaconsistentPoint(**{**p.__dict__, "d_penalized": Value.missing()})
    v = RankingView(_Repo([p, _pt("y", 0.5)]))
    col = _COLS.index("d_penalized")
    assert v._table.item(0, _COLS.index("run_tag")).text() == "y"
    assert v._table.item(1, col).text() == "—"


def test_double_click_emits_run(qapp):
    v = RankingView(_Repo([_pt("run1", 0.5)]))
    seen = []
    v.run_activated.connect(lambda e, t: seen.append((e, t)))
    v._on_double_click(v._table.item(0, 0))
    assert seen == [("thesis", "run1")]
