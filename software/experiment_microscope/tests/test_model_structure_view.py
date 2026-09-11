"""Model Structure tab: the checkpoint's own topology, independent of any
window's activations (FIXME §15, §18, §19)."""

import numpy as np

from experiment_microscope.processing.meeting01 import AeTrace, EncoderLayer
from experiment_microscope.views.model_structure_view import ModelStructureView


def _trace() -> AeTrace:
    enc = (
        EncoderLayer(kind="linear", output=np.zeros(64), weight=np.zeros((64, 256))),
        EncoderLayer(kind="lif", output=np.zeros(64), v_mem=np.zeros(64), voltage_threshold=1.05),
        EncoderLayer(kind="linear", output=np.zeros(32), weight=np.zeros((32, 64))),
    )
    dec = (
        EncoderLayer(kind="linear", output=np.zeros(64), weight=np.zeros((64, 32))),
        EncoderLayer(kind="lif", output=np.zeros(64), v_mem=np.zeros(64), voltage_threshold=1.09),
        EncoderLayer(kind="linear", output=np.zeros(256), weight=np.zeros((256, 64))),
    )
    return AeTrace(latent=np.zeros(32), reconstruction=np.zeros(256),
                    encoded_input=np.zeros(256), encoder_layers=enc, decoder_layers=dec)


def test_show_trace_builds_layer_table(qapp):
    v = ModelStructureView()
    spec = {"architecture": "conv1d", "encoding": "latency", "dataset": "fsdd",
            "fold": 2, "role": "final", "run": 3}
    v.show_trace(_trace(), spec)

    assert v._table.rowCount() == 6  # 3 encoder + 3 decoder layers
    assert v._table.item(0, 1).text() == "linear"
    assert v._table.item(0, 2).text() == "256 → 64"
    assert v._table.item(0, 3).text() == f"{64 * 256:,}"
    assert v._table.item(1, 1).text() == "lif"
    assert "1.05" in v._table.item(1, 4).text()

    header = v._header.text()
    assert "conv1d/latency" in header
    assert "256 → 64 → 64 → 32 → 64 → 64 → 256" in header
    total_params = 64 * 256 + 32 * 64 + 64 * 32 + 256 * 64
    assert f"{total_params:,}" in header


def test_empty_state_before_any_trace(qapp):
    v = ModelStructureView()
    assert v._table.rowCount() == 0
    assert "meeting01" in v._header.text() or "meeting01" in v._header.text().lower()
