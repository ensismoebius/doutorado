"""Model Structure — the selected model's whole topology, on its own (FIXME §15,
§18, §19).

Where the "Autoencoder" tab draws the network *doing work* (activations for one
window), this tab shows the network *as an object*: every layer, its shape, its
parameter count, and — for a spiking layer — its trained firing threshold. It
never depends on the input signal, only on the checkpoint that was loaded, so
two different windows through the same model always show the same structure.

It mirrors whatever model is currently loaded in the Autoencoder tab
(``AutoencoderView.trace_changed``) — browsing to a new window updates it
automatically, and picking a different model there ("Run this model") updates
it just the same way.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from experiment_microscope.core.i18n import t as _t
from experiment_microscope.views._help import HelpBox

_HELP = """
<b>What this panel shows.</b> The selected model's whole <b>structure</b> — every
layer, its shape, how many trainable weights it has, and (for a spiking layer)
its trained firing threshold. This is a property of the <b>checkpoint</b>, not of
any particular window: it never changes when you browse to a different sample.
<br><br>
<b>Structure vs. activation</b> — confusable pair: this tab is the blueprint (the
same for every window); the "Autoencoder" tab is the blueprint <i>at work</i>
(activations differ per window). Use this tab to answer "how big / how deep is
this network", the other to answer "what did it do with this signal".
<br><br>
It follows whichever model is loaded in the "Autoencoder" tab — browse a window
there, or pick a specific trained model and press <i>Run this model</i>, and this
table updates the same way.
"""


class ModelStructureView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.addWidget(HelpBox("Model Structure", _HELP))
        self._header = QLabel(_t("Select a meeting01 window (Autoencoder tab) to load a model."))
        self._header.setWordWrap(True)
        root.addWidget(self._header)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            _t("half"), _t("layer"), _t("shape (in → out)"), _t("trainable weights"), _t("extra"),
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self._table, 1)

    # -- external API --------------------------------------------------
    def show_trace(self, trace, spec: dict) -> None:
        """Called whenever ``AutoencoderView`` loads a fresh trace — auto
        (browsing) or explicit ("Run this model")."""
        n_enc = len(trace.encoder_layers)
        sizes = [int(np.asarray(trace.encoded_input).size)]
        rows: list[tuple[str, str, str, int, str]] = []
        total_params = 0
        for j, lyr in enumerate((*trace.encoder_layers, *trace.decoder_layers)):
            half = _t("Encoder") if j < n_enc else _t("Decoder")
            out_n = None if lyr.output is None else int(np.asarray(lyr.output).size)
            if out_n is not None:
                sizes.append(out_n)
            if lyr.weight is not None:
                w = np.asarray(lyr.weight)
                params = int(w.size)
                total_params += params
                shape_txt = f"{w.shape[1]} → {w.shape[0]}"
            else:
                params = 0
                shape_txt = str(out_n) if out_n is not None else "—"
            extra = (_t("v_th = {v}", v=f"{lyr.voltage_threshold:.4g}")
                     if lyr.voltage_threshold is not None else "—")
            rows.append((half, lyr.kind, shape_txt, params, extra))

        self._table.setRowCount(len(rows))
        for r, (half, kind, shape_txt, params, extra) in enumerate(rows):
            for c, val in enumerate((half, kind, shape_txt, f"{params:,}" if params else "—", extra)):
                item = QTableWidgetItem(str(val))
                self._table.setItem(r, c, item)

        neurons = " → ".join(str(s) for s in sizes)
        self._header.setText(_t(
            "{arch}/{enc} · dataset {ds} fold {fold} · {role} run {run}\n"
            "{neurons} neurons · {params} trainable weights (Linear only — bias not exposed by the binding)",
            arch=spec.get("architecture", "?"), enc=spec.get("encoding", "?"),
            ds=spec.get("dataset", "?"), fold=spec.get("fold", "?"),
            role=spec.get("role", "?"), run=spec.get("run", "?"),
            neurons=neurons, params=f"{total_params:,}",
        ))
