"""Raw signal viewer dock (FIXME §8, §25, §29).

For milestone 1 this renders whatever ``adapter.load_signal(node)`` returns. For
meeting01 / thesis that requires the ``nn_microscope`` binding (signals are
recomputed, not persisted); until it is built the panel shows the exact build
command rather than an empty plot (no-fallback).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg


class SignalView(QWidget):
    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if PG_OK:
            self._plot = pg.PlotWidget()
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("bottom", "sample")
            self._plot.addLegend()
            layout.addWidget(self._plot)
            self._message = pg.LabelItem(justify="left")
        else:
            self._plot = None
            layout.addWidget(missing_widget("Signal view"))

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        if self._plot is None:
            return
        self._plot.clear()
        adapter = self.repo.adapter(adapter_key)
        try:
            signal: Signal1D = adapter.load_signal(node)
        except NotImplementedError:
            self._banner("This object has no raw signal.")
            return
        except BindingUnavailableError as exc:
            self._banner(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._banner(f"load_signal failed: {exc}")
            return
        self._render(signal)

    def _banner(self, text: str) -> None:
        self._plot.clear()
        item = pg.TextItem(text, anchor=(0, 0), color=(200, 200, 200))
        self._plot.addItem(item)
        item.setPos(0, 0)

    def _render(self, signal: Signal1D) -> None:
        data = np.asarray(signal.samples)
        if data.ndim == 1:
            data = data[None, :]
        names = signal.channel_names or tuple(f"ch{i}" for i in range(data.shape[0]))
        for i, row in enumerate(data):
            self._plot.plot(
                np.arange(row.shape[0]),
                row + i * 0.0,
                pen=pg.intColor(i, hues=max(3, data.shape[0])),
                name=names[i] if i < len(names) else f"ch{i}",
            )
        title = signal.label or ""
        if signal.origin is not None:
            title = f"{title}  [{signal.origin.value}]".strip()
        self._plot.setTitle(title)
        self._plot.setLabel("left", signal.unit or "amplitude")
