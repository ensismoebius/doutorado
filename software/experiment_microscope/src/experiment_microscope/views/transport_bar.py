"""Shared animation transport (FIXME §24, §25).

One strip of controls bound to the single :class:`TimelinePlayer`. Whatever
temporal view is active registers its frame count with the player; this bar
drives Play / Pause / Stop / Step / Loop / Speed and shows the frame position.
The bar never touches view data — it only moves the frame index.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from experiment_microscope.core.animation import SPEED_PRESETS, TimelinePlayer


class TransportBar(QWidget):
    def __init__(self, player: TimelinePlayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._player = player

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        self._play = QPushButton("▶")
        self._play.setCheckable(True)
        self._play.setFixedWidth(32)
        self._play.toggled.connect(self._on_play_toggled)
        self._stop = QPushButton("⏹")
        self._stop.setFixedWidth(32)
        self._stop.clicked.connect(player.stop)
        self._back = QPushButton("⏮")
        self._back.setFixedWidth(32)
        self._back.clicked.connect(player.step_backward)
        self._fwd = QPushButton("⏭")
        self._fwd.setFixedWidth(32)
        self._fwd.clicked.connect(player.step_forward)
        for w in (self._back, self._play, self._fwd, self._stop):
            row.addWidget(w)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderMoved.connect(player.seek)
        row.addWidget(self._slider, 1)

        self._pos = QLabel("—")
        self._pos.setMinimumWidth(72)
        row.addWidget(self._pos)

        self._loop = QPushButton("loop")
        self._loop.setCheckable(True)
        self._loop.toggled.connect(player.set_looping)
        row.addWidget(self._loop)

        self._speed = QComboBox()
        for s in SPEED_PRESETS:
            self._speed.addItem(f"{s:g}×", s)
        self._speed.setCurrentIndex(SPEED_PRESETS.index(1.0))
        self._speed.currentIndexChanged.connect(
            lambda: player.set_speed(self._speed.currentData())
        )
        row.addWidget(self._speed)

        player.frame_changed.connect(self._on_frame)
        player.range_changed.connect(self._on_range)
        player.playing_changed.connect(self._on_playing)
        self._on_range(player.total_frames)
        self._on_frame(player.current_frame)

    # -- player -> UI -------------------------------------------
    def _on_frame(self, frame: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._pos.setText(f"{frame} / {max(0, self._player.total_frames - 1)}")

    def _on_range(self, total: int) -> None:
        self._slider.setRange(0, max(0, total - 1))
        self._on_frame(self._player.current_frame)
        self.setEnabled(total > 1)

    def _on_playing(self, playing: bool) -> None:
        self._play.blockSignals(True)
        self._play.setChecked(playing)
        self._play.setText("⏸" if playing else "▶")
        self._play.blockSignals(False)

    # -- UI -> player ------------------------------------------
    def _on_play_toggled(self, checked: bool) -> None:
        self._player.play() if checked else self._player.pause()
