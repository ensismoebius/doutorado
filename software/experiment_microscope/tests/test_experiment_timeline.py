import types

import pytest

pytest.importorskip("PySide6")

from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.experiment_timeline import ExperimentTimeline


def _cfg(**kw):
    base = dict(config_id="c", dataset="fsdd", fold=0, status="done",
                best_val=None, best_epoch=None, epochs=[])
    base.update(kw)
    return types.SimpleNamespace(**base)


class _FakeState:
    def __init__(self, configs):
        self.session = {"git_commit": "abcdef1234567", "seed": 42}
        self.configs = {c.config_id: c for c in configs}
        self.folds_seen = {(c.dataset, c.fold) for c in configs}


class _Adapter:
    def __init__(self, state):
        self._state = state

    def session_state(self):
        return self._state


def _repo(state):
    r = DataRepository()
    r._adapters["meeting01"] = _Adapter(state)
    return r


def test_empty_stream_shows_reason(qapp):
    v = ExperimentTimeline(_repo(None))
    v.refresh()
    assert v._tree.topLevelItemCount() == 0
    assert "No meeting01 event stream" in v._status.text()


def test_builds_fold_config_epoch_tree(qapp):
    state = _FakeState([
        _cfg(config_id="snn-ae_direct_seed42_run1", fold=0, best_val=0.0123, best_epoch=7,
             epochs=[(1, 0.09, 0.08), (2, 0.05, 0.06)]),
        _cfg(config_id="lstm-ae_direct_seed42_run1", fold=1, status="running"),
    ])
    v = ExperimentTimeline(_repo(state))
    v.refresh()
    assert v._tree.topLevelItemCount() == 2                 # fold 0, fold 1
    fold0 = v._tree.topLevelItem(0)
    cfg0 = fold0.child(0)
    assert "best val 0.0123 @ epoch 7" in cfg0.text(1)
    assert cfg0.childCount() == 2                           # 2 epochs
    assert cfg0.child(0).text(1) == "train 0.09   val 0.08"


def test_activating_config_emits_payload(qapp):
    state = _FakeState([_cfg(config_id="k", fold=3, dataset="audiomnist")])
    v = ExperimentTimeline(_repo(state))
    v.refresh()
    seen = []
    v.config_activated.connect(lambda d, f, c: seen.append((d, f, c)))
    fold_item = v._tree.topLevelItem(0)
    v._on_activated(fold_item.child(0), 0)
    assert seen == [("audiomnist", 3, "k")]
