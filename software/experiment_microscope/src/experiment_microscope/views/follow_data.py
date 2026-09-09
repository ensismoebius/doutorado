"""The "Follow the Data" strip (FIXME §9, §55).

A horizontal chain of pipeline stages. For the current selection each stage is
either **available** (the adapter can produce it — clickable, opens the matching
view) or **absent** (dimmed, with a one-line reason on hover). This is the
app's spine: it makes the transformation chain visible at a glance and lets the
researcher step ← upstream / downstream → without losing the selected object.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    tab: str  # central-tab name this stage opens, or "" if none yet


STAGES: tuple[Stage, ...] = (
    Stage("raw", "RAW", "Signal"),
    Stage("window", "WINDOW", "Signal"),
    Stage("normalized", "NORMALIZED", "Signal"),
    Stage("wavelet", "WAVELET", "Wavelet Lab"),
    Stage("features", "FEATURES", "Feature Matrix"),
    Stage("paraconsistent", "PARACONSISTENT", "Paraconsistent plane"),
    Stage("latent", "LATENT", ""),
    Stage("reconstruction", "RECONSTRUCTION", ""),
    Stage("classification", "CLASSIFICATION", ""),
)


class FollowDataBar(QWidget):
    #: emitted with a central-tab name when a stage is clicked
    stage_activated = Signal(str)

    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._buttons: dict[str, QPushButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(2)
        for i, stage in enumerate(STAGES):
            if i:
                arrow = QLabel("→")
                arrow.setEnabled(False)
                layout.addWidget(arrow)
            btn = QPushButton(stage.label)
            btn.setFlat(True)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _=False, s=stage: self._on_click(s))
            self._buttons[stage.key] = btn
            layout.addWidget(btn)
        layout.addStretch(1)

    def _on_click(self, stage: Stage) -> None:
        if stage.tab:
            self.stage_activated.emit(stage.tab)

    def update_for(self, node: TreeNode | None, adapter_key: str | None) -> None:
        available = self._availability(node, adapter_key)
        for stage in STAGES:
            btn = self._buttons[stage.key]
            reason = available.get(stage.key, "not applicable to this selection")
            ok = reason == "" and bool(stage.tab)
            btn.setEnabled(ok)
            btn.setToolTip("" if ok else reason)
            font = btn.font()
            font.setBold(ok)
            btn.setFont(font)

    def _availability(self, node: TreeNode | None, adapter_key: str | None) -> dict[str, str]:
        """Map stage_key -> "" (available) or a short reason string."""
        out = {s.key: "no selection" for s in STAGES}
        if node is None or adapter_key is None:
            return out
        adapter = self.repo.adapter(adapter_key)
        h = getattr(node, "handle", {}) or {}
        level = h.get("level")

        is_signal = level in ("window", "sample")
        if is_signal:
            out["raw"] = out["normalized"] = ""
            out["window"] = "" if level == "window" else "this sample is not windowed here"
            out["wavelet"] = ""
        else:
            for k in ("raw", "window", "normalized", "wavelet"):
                out[k] = "select an individual window / sample"

        if adapter_key == "thesis" and level == "run":
            out["features"] = ""
        elif adapter_key == "thesis" and is_signal:
            out["features"] = "select the run node to recompute its feature matrix"
        else:
            out["features"] = "handcrafted-feature matrix is a thesis-run view"

        # the paraconsistent plane is always populated (all persisted points)
        out["paraconsistent"] = ""

        for k in ("latent", "reconstruction", "classification"):
            out[k] = "needs a trained model .npz (not yet available)"
        return out
