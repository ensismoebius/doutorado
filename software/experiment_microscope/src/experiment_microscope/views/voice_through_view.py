"""Voice Through the Network — play a whole recording through the trained
SNN-AE, one window per animation frame (companion to Autoencoder, FIXME §15/§18).

Every other SNN-AE view (Autoencoder, SNN Lab, Reconstruction) answers "what
does the network do on ONE 256-sample window" — a single frozen instant. This
answers the question a recording actually poses: what does the network do as
a whole spoken digit passes through it, window after window, in true
recording order? "True order" matters — the split arrays interleave windows
from many different recordings (shuffling is part of what makes a split a
split), so grouping by ``recording_id`` and sorting by
``source_window_index`` (``Meeting01Adapter.recordings_for``) is what turns a
scattered bag of windows back into one utterance to play.

No second, subtly different neuron-graph implementation lives here: this view
embeds ``AutoencoderView`` itself and drives it one frame at a time via
``AutoencoderView.show_frame`` — a window's activation picture looks
identical whether you reached it by ordinary browsing or by pressing ▶ here.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.core.i18n import t as _t
from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.views._help import HelpBox
from experiment_microscope.views.autoencoder_view import AutoencoderView

#: a combo with hundreds of recordings is unusable to scroll — cap the list
#: the same way the Data Explorer caps its window list (FIXME §7).
_MAX_RECORDINGS = 200

_HELP = """
<b>What this shows.</b> The same neuron graph as the Autoencoder tab, but
fed a whole <b>recording</b> instead of one window — press ▶ and watch the
network's activity change as an entire spoken digit passes through it,
window by window, in the order it was actually spoken.
<br><br>
<b>Controls.</b> <i>split</i> / <i>recording</i> pick which utterance to
play (grouped from the same windows every other view uses, just put back in
time order). <i>model</i> picks which trained checkpoint runs it — chosen
once, then held fixed for the whole recording, so what changes frame to
frame is only the input, never the model.
<br><br>
<b>Reading it.</b> Node colour/size still means activation strength, same as
the Autoencoder tab — click a neuron there for its exact number. A short
recording may be only 2–4 windows; the transport's frame count updates to
match whichever recording is selected.
"""


class VoiceThroughView(QWidget):
    def __init__(self, repo, app_state=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.app_state = app_state
        self._player = None
        self._node: TreeNode | None = None
        self._recordings: list[dict] = []
        self._specs: list[dict] = []
        self._current_rows: list[int] = []
        self._current_spec: dict | None = None

        root = QVBoxLayout(self)
        root.addWidget(HelpBox("Voice Through the Network", _HELP))

        bar = QHBoxLayout()
        bar.addWidget(QLabel(_t("split")))
        self._split_combo = QComboBox()
        self._split_combo.addItems(["test", "val", "train"])
        self._split_combo.currentIndexChanged.connect(self._reload_recordings)
        bar.addWidget(self._split_combo)
        bar.addWidget(QLabel(_t("recording")))
        self._rec_combo = QComboBox()
        self._rec_combo.setMinimumWidth(260)
        self._rec_combo.currentIndexChanged.connect(self._on_recording_changed)
        bar.addWidget(self._rec_combo, 1)
        bar.addWidget(QLabel(_t("model")))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(220)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        bar.addWidget(self._model_combo)
        root.addLayout(bar)

        self._status = QLabel(_t("Select a meeting01 fold or window first."))
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._ae = AutoencoderView(repo, selection=None, app_state=app_state)
        self._ae.set_controls_visible(False)
        # deliberately NOT set_timeline()'d — an embedded instance must never
        # reset ITS OWN flood-column player from _load_trace(); this view is
        # the only thing allowed to decide what "frame" means here (a window,
        # not a layer). See AutoencoderView.show_frame's docstring.
        root.addWidget(self._ae, 1)

    # -- external API -------------------------------------------
    def set_timeline(self, player) -> None:
        self._player = player
        player.frame_changed.connect(self._on_frame)

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        h = getattr(node, "handle", {}) or {}
        if adapter_key != "meeting01" or h.get("dataset") is None or h.get("cv_fold") is None:
            self._node = None
            self._current_rows = []
            self._status.setText(_t("Select a meeting01 fold or window first."))
            self._rec_combo.clear()
            self._model_combo.clear()
            self._ae.show_node(TreeNode("x", "", {"level": "root"}), adapter_key)
            return
        self._node = node
        if h.get("split"):
            i = self._split_combo.findText(h["split"])
            if i >= 0:
                self._split_combo.blockSignals(True)
                self._split_combo.setCurrentIndex(i)
                self._split_combo.blockSignals(False)
        self._reload_recordings()
        self._populate_models()

    # -- internals ------------------------------------------------
    def _reload_recordings(self) -> None:
        if self._node is None:
            return
        adapter = self.repo.adapter("meeting01")
        h = dict(self._node.handle)
        h["split"] = self._split_combo.currentText()
        probe = TreeNode(self._node.kind, self._node.label, h)
        try:
            self._recordings = adapter.recordings_for(probe)
        except Exception as exc:  # noqa: BLE001
            self._recordings = []
            self._status.setText(_t("recordings_for failed: {exc}", exc=exc))
            return
        self._rec_combo.blockSignals(True)
        self._rec_combo.clear()
        for r in self._recordings[:_MAX_RECORDINGS]:
            self._rec_combo.addItem(_t(
                "recording {rid} — speaker {spk}, digit {d}, {n} windows",
                rid=r["recording_id"], spk=r["speaker"], d=r["digit"], n=len(r["rows"])))
        self._rec_combo.blockSignals(False)
        self._rec_combo.setEnabled(bool(self._recordings))
        if self._recordings:
            self._on_recording_changed(0)
        else:
            self._current_rows = []
            self._status.setText(_t("No recordings in this split."))

    def _populate_models(self) -> None:
        if self._node is None:
            return
        adapter = self.repo.adapter("meeting01")
        try:
            self._specs = adapter.models_for(self._node)
        except Exception:  # noqa: BLE001
            self._specs = []
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for s in self._specs:
            self._model_combo.addItem(_t(
                "{role} · {arch}/{enc} · run {run}",
                role=s.get("role", "?"), arch=s["architecture"], enc=s["encoding"],
                run=s.get("run", "?")))
        self._model_combo.blockSignals(False)
        self._model_combo.setEnabled(bool(self._specs))
        if self._specs:
            self._model_combo.setCurrentIndex(0)
            self._on_model_changed(0)
        else:
            self._current_spec = None

    def _on_model_changed(self, i: int) -> None:
        self._current_spec = self._specs[i] if 0 <= i < len(self._specs) else None
        self._render_current_frame()

    def _on_recording_changed(self, i: int) -> None:
        if not (0 <= i < len(self._recordings)):
            self._current_rows = []
            return
        self._current_rows = self._recordings[i]["rows"]
        if self._player is not None:
            self._player.set_total_frames(max(1, len(self._current_rows)))
            self._player.seek(0)
        self._render_current_frame(0)

    def _on_frame(self, frame: int) -> None:
        if self._current_rows:
            self._render_current_frame(frame)

    def _render_current_frame(self, frame: int | None = None) -> None:
        if not self._current_rows or self._current_spec is None or self._node is None:
            return
        k = frame if frame is not None else (self._player.current_frame if self._player else 0)
        k = max(0, min(len(self._current_rows) - 1, k))
        h = dict(self._node.handle)
        h.update(level="window", split=self._split_combo.currentText(), row=self._current_rows[k])
        node = TreeNode("sample", _t("window {k}", k=k), h)
        self._ae.show_frame(node, self._current_spec)
        ri = self._rec_combo.currentIndex()
        rec = self._recordings[ri] if 0 <= ri < len(self._recordings) else None
        if rec is not None:
            self._status.setText(_t(
                "digit {d}, speaker {spk} — window {k} / {n} — press ▶ to play "
                "the whole utterance through the network",
                d=rec["digit"], spk=rec["speaker"], k=k + 1, n=len(self._current_rows)))
