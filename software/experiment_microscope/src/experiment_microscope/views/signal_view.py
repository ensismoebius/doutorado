"""Raw signal viewer dock (FIXME §8, §25, §29).

For milestone 1 this renders whatever ``adapter.load_signal(node)`` returns. For
meeting01 / thesis that requires the ``nn_microscope`` binding (signals are
recomputed, not persisted); until it is built the panel shows the exact build
command rather than an empty plot (no-fallback).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from experiment_microscope.core.selection import SelectionState
from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg
from experiment_microscope.views._timesync import TimeCursor


class SignalView(QWidget):
    def __init__(
        self,
        repo: DataRepository,
        selection: SelectionState | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cursor = None
        self._last_signal = None
        if PG_OK:
            self._plot = pg.PlotWidget()
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("bottom", "sample")
            self._plot.addLegend()
            layout.addWidget(self._plot)
            self._message = pg.LabelItem(justify="left")
            if selection is not None:
                self._cursor = TimeCursor(self._plot.getPlotItem(), selection)
        else:
            self._plot = None
            layout.addWidget(missing_widget("Signal view"))

    def can_export(self) -> bool:
        return getattr(self, "_last_signal", None) is not None

    def export_figure(self, path, **opts):
        from experiment_microscope.viz.mpl_export import annotate_provenance, new_figure, save_figure

        sig = self._last_signal
        data = np.asarray(sig.samples)
        if data.ndim == 1:
            data = data[None, :]
        fig = new_figure(**{k: opts[k] for k in ("width_in", "height_in", "dpi") if k in opts})
        ax = fig.add_subplot(111)
        names = sig.channel_names or tuple(f"ch{i}" for i in range(data.shape[0]))
        for i, row in enumerate(data):
            ax.plot(row, lw=0.8, label=names[i] if i < len(names) else f"ch{i}")
        ax.set_xlabel("sample")
        ax.set_ylabel(sig.unit or "amplitude")
        ax.set_title(sig.label or "signal")
        if data.shape[0] > 1:
            ax.legend(fontsize=6, ncol=min(6, data.shape[0]))
        annotate_provenance(ax, f"origin: {sig.origin.value}")
        return save_figure(fig, path, transparent=opts.get("transparent", False))

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
        self._last_signal = signal
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
        if self._cursor is not None:
            self._cursor.reattach()
