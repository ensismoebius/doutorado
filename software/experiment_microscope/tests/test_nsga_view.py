import json

import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.data.paraconsistent_ga_adapter import ParaconsistentGaAdapter
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.nsga_view import NsgaView

_IND = {
    "rank": 0, "feasible": True, "d_truth": 1.84, "d_penalized_mean": 1.88,
    "d_penalized_std": 0.008, "alpha": 0.05, "beta": 0.9, "g1": -0.85, "g2": -0.04,
    "inference_cost": 256, "param_count": 769, "est_latency_ms": 0.000512,
    "latent_activity": 1.09, "born_generation": 21, "constraint_violation": 0.0,
    "genome": {"depth": 1, "encoding": "direct", "latent": 1, "time_steps": 1,
               "voltage_threshold": 1.0},
}
_INFEASIBLE = {**_IND, "rank": 3, "feasible": False, "d_penalized_mean": 2.5,
               "inference_cost": 90, "param_count": 100, "genome": {"latent": 9, "depth": 2}}


def _adapter(tmp_path):
    payload = {
        "run_tag": "pga_demo", "n_evaluated": 512,
        "pareto_front": [_IND, _INFEASIBLE],
        "population": {"modality": "eeg", "model": "ann-ae"},  # run metadata, not individuals
        "ga": {"generations": 64, "population_size": 32},
        "constraints": {}, "warnings": ["est latency UNCALIBRATED"],
    }
    (tmp_path / "pga_demo_pareto.json").write_text(json.dumps(payload))
    return ParaconsistentGaAdapter(results_dir=tmp_path)


def _repo(tmp_path):
    r = DataRepository()
    r._adapters["paraconsistent_ga"] = _adapter(tmp_path)
    return r


def test_loads_run_and_reports_front(qapp, tmp_path):
    v = NsgaView(_repo(tmp_path))
    assert v._run.currentText() == "pga_demo"
    assert "2 Pareto-front individual(s)" in v._status.text()
    assert "UNCALIBRATED" in v._status.text()
    assert "512 evaluated" in v._status.text()


def test_front_points_plotted(qapp, tmp_path):
    v = NsgaView(_repo(tmp_path))
    assert len(v._scatter.points()) == 2


def test_click_shows_genome_and_uncalibrated_latency(qapp, tmp_path):
    v = NsgaView(_repo(tmp_path))
    v._on_click(v._scatter, v._scatter.points()[:1])
    labels = {v._table.item(r, 0).text(): v._table.item(r, 1).text()
              for r in range(v._table.rowCount())}
    assert "UNCALIBRATED" in labels["est_latency_ms"]
    assert "genome.encoding" in labels


def test_no_results_message(qapp, tmp_path):
    r = DataRepository()
    r._adapters["paraconsistent_ga"] = ParaconsistentGaAdapter(results_dir=tmp_path / "empty")
    v = NsgaView(r)
    assert "No paraconsistentGA results" in v._status.text()
