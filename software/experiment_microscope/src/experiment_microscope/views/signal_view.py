"""Raw signal viewer dock (FIXME §8, §25, §29).

For milestone 1 this renders whatever ``adapter.load_signal(node)`` returns. For
meeting01 / thesis that requires the ``nn_microscope`` binding (signals are
recomputed, not persisted); until it is built the panel shows the exact build
command rather than an empty plot (no-fallback).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from experiment_microscope.core.selection import SelectionState
from experiment_microscope.views._help import HelpBox
from experiment_microscope.viz.audio import AudioPlayer, is_available as _audio_ok
from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg
from experiment_microscope.views._plotinfo import HoverReadout, autofit, fade_in, set_source
from experiment_microscope.views._timesync import TimeCursor


_HELP = """
<b>What this shows.</b> The signal exactly as the pipeline sees it for the
selected sample — an audio waveform, or several stacked EEG channels
(EEG = electroencephalogram, brain electrical activity).
<br><br>
<b>Axes.</b> Horizontal = sample number (multiply by 1/sampling-rate for seconds;
the rate is in the title). Vertical = amplitude. For meeting01 windows the
amplitude is <b>z-scored</b> (mean 0, spread 1) because that is what the models
receive; the unit is shown on the left axis.
<br><br>
<b>Colours / lines.</b> One colour per channel, named in the legend. Drag on the
plot to select a time range — downstream views can restrict to it. The vertical
cursor line reports the exact value under it.
<br><br>
<b>DISPLAY-DOWNSAMPLED</b> in the title (and the status bar) means the drawn
curve is decimated for speed; the cursor still reads the full-resolution number.
<br><br>
<b>🔊 Listen</b> plays the waveform through your speakers (audio samples only;
EEG has no sound). The <b>▶</b> transport at the bottom is separate — it steps
animation frames, it does not play audio.
"""



class SignalView(QWidget):
    #: above this many samples per channel the plot decimates for display and
    #: the status bar shows DISPLAY-DOWNSAMPLED (FIXME §26). Exact values are
    #: still readable via the cursor because the full array is kept in
    #: ``_last_signal``.
    DISPLAY_LIMIT = 20_000

    def __init__(
        self,
        repo: DataRepository,
        selection: SelectionState | None = None,
        app_state=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        self.app_state = app_state
        layout = QVBoxLayout(self)
        layout.addWidget(HelpBox('Raw signal', _HELP))
        layout.setContentsMargins(0, 0, 0, 0)
        self._cursor = None
        self._last_signal = None
        self._audio = AudioPlayer(self)

        bar = QHBoxLayout()
        from experiment_microscope.core.i18n import t as _t
        self._listen = QPushButton(_t("\N{SPEAKER WITH THREE SOUND WAVES}  Listen"))
        self._listen.setToolTip(
            "Play this waveform through the default audio output. The \N{BLACK RIGHT-POINTING TRIANGLE} "
            "transport below only steps animation frames — it is not sound."
        )
        self._listen.clicked.connect(self._on_listen)
        self._listen.setEnabled(False)
        self._audio_note = QLabel("")
        self._audio_note.setWordWrap(True)
        bar.addWidget(self._listen)
        bar.addWidget(self._audio_note, 1)
        layout.addLayout(bar)

        if PG_OK:
            self._plot = pg.PlotWidget()
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("bottom", "sample")
            self._plot.addLegend()
            layout.addWidget(self._plot)
            self._message = pg.LabelItem(justify="left")
            self._hover = HoverReadout(self._plot, x_label="sample")
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
        self._adapter_key = adapter_key
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

    def _is_audio(self, signal: Signal1D) -> bool:
        data = np.asarray(signal.samples)
        return data.ndim == 1 and np.isfinite(signal.sample_rate) and signal.sample_rate >= 3000.0

    def _on_listen(self) -> None:
        sig = self._last_signal
        if sig is None:
            return
        try:
            dur = self._audio.play(np.asarray(sig.samples).reshape(-1), sig.sample_rate)
        except RuntimeError as exc:
            self._audio_note.setText(str(exc))
            return
        norm = "" if (sig.unit or "").lower() not in ("z-score", "") else "  · amplitude peak-normalised for listening"
        self._audio_note.setText(
            f"playing {dur * 1000:.0f} ms at {sig.sample_rate:.0f} Hz{norm}"
        )

    def _banner(self, text: str) -> None:
        self._listen.setEnabled(False)
        self._audio_note.setText("")
        self._plot.clear()
        item = pg.TextItem(text, anchor=(0, 0), color=(200, 200, 200))
        self._plot.addItem(item)
        item.setPos(0, 0)

    def _render(self, signal: Signal1D) -> None:
        self._last_signal = signal
        audio = self._is_audio(signal)
        self._listen.setEnabled(audio)
        if not audio:
            self._audio_note.setText(
                "not audio (multi-channel or sub-3kHz) — nothing to play"
                if np.asarray(signal.samples).ndim > 1 or signal.sample_rate < 3000.0
                else ""
            )
        elif not _audio_ok():
            self._listen.setEnabled(False)
            self._audio_note.setText("QtMultimedia not installed — see Listen tooltip")
        else:
            self._audio_note.setText("")
        data = np.asarray(signal.samples)
        if data.ndim == 1:
            data = data[None, :]
        names = signal.channel_names or tuple(f"ch{i}" for i in range(data.shape[0]))
        n = data.shape[1]
        stride = max(1, -(-n // self.DISPLAY_LIMIT))  # ceil(n / limit)
        downsampled = stride > 1
        x = np.arange(0, n, stride)
        for i, row in enumerate(data):
            from experiment_microscope.core import palette

            single = data.shape[0] == 1
            self._plot.plot(
                x,
                row[::stride],
                pen=(palette.pen("input", 2) if single
                     else pg.intColor(i, hues=max(3, data.shape[0]))),
                name=names[i] if i < len(names) else f"ch{i}",
            )
        if self.app_state is not None:
            self.app_state.display_downsampled = downsampled
        title = signal.label or ""
        if signal.origin is not None:
            title = f"{title}  [{signal.origin.value}]".strip()
        if downsampled:
            title += f"  · DISPLAY-DOWNSAMPLED 1:{stride} ({n:,}→{x.size:,} pts; cursor reads full-res)"
        self._plot.setTitle(title)
        self._plot.setLabel("left", signal.unit or "amplitude")
        set_source(
            self._plot,
            f"{getattr(self, '_adapter_key', '?')}.load_signal() · "
            f"origin [{signal.origin.value if signal.origin else 'unknown'}] · "
            f"{signal.sample_rate:.0f} Hz",
        )
        if getattr(self, "_hover", None) is not None:
            self._hover.reattach()
        if self._cursor is not None:
            self._cursor.reattach()
        autofit(self._plot)
        fade_in(self._plot)
