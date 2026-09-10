"""The Guided Tour dock (didactic redesign).

A big, readable panel on the left. It walks the audience through one pipeline a
single step at a time: a plain-language paragraph, the one number that matters,
and Back / Next. Each step drives the rest of the app — selects the right data,
opens the right tab — so the teacher only has to talk and click Next.

"Free explore" dismisses the tour and hands the app back to the user (the wide
mouth of Segel & Heer's martini glass).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.core.story import STORIES, Story


class StoryPanel(QWidget):
    finished = Signal()

    def __init__(self, workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ws = workspace
        self._story: Story | None = None
        self._i = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._story_title = QLabel()
        self._story_title.setWordWrap(True)
        self._story_title.setStyleSheet("font-size:12px;color:#888;")
        root.addWidget(self._story_title)

        self._dots = QLabel()
        self._dots.setStyleSheet("font-size:16px;letter-spacing:3px;")
        root.addWidget(self._dots)

        self._step_title = QLabel()
        self._step_title.setWordWrap(True)
        self._step_title.setStyleSheet("font-size:16px;font-weight:bold;")
        root.addWidget(self._step_title)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.TextFormat.RichText)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._body.setStyleSheet("font-size:13px;line-height:150%;")
        area.setWidget(self._body)
        root.addWidget(area, 1)

        self._callout = QLabel()
        self._callout.setWordWrap(True)
        self._callout.setTextFormat(Qt.TextFormat.RichText)
        self._callout.setFrameShape(QFrame.Shape.StyledPanel)
        self._callout.setMargin(8)
        self._callout.setStyleSheet(
            "background:#2d3a2d;border:1px solid #4a5a4a;font-size:13px;")
        root.addWidget(self._callout)

        nav = QHBoxLayout()
        self._back = QPushButton("◀ Back")
        self._back.clicked.connect(self._prev)
        self._next = QPushButton("Next ▶")
        self._next.clicked.connect(self._advance)
        self._exit = QPushButton("Free explore")
        self._exit.clicked.connect(self._finish)
        nav.addWidget(self._back)
        nav.addWidget(self._next)
        nav.addStretch(1)
        nav.addWidget(self._exit)
        root.addLayout(nav)

    # -- public -----------------------------------------------------
    def start(self, key: str) -> None:
        self._story = STORIES.get(key) or STORIES["meeting01"]
        self._story_title.setText(f"GUIDED TOUR — {self._story.title}")
        self._i = 0
        self._render()

    # -- navigation ----------------------------------------------
    def _prev(self) -> None:
        if self._i > 0:
            self._i -= 1
            self._render()

    def _advance(self) -> None:
        if self._story is None:
            return
        if self._i >= len(self._story.steps) - 1:
            self._finish()
            return
        self._i += 1
        self._render()

    def _finish(self) -> None:
        self.finished.emit()

    # -- rendering ---------------------------------------------------
    def _render(self) -> None:
        if self._story is None:
            return
        n = len(self._story.steps)
        step = self._story.steps[self._i]
        self._dots.setText("".join("●" if j == self._i else "○" for j in range(n)))
        self._step_title.setText(step.title)
        self._body.setText(step.narration)
        self._back.setEnabled(self._i > 0)
        self._next.setText("Finish ▶" if self._i == n - 1 else "Next ▶")

        # drive the app: select data, open the tab, run any extra action
        try:
            if step.select is not None:
                picked = step.select(self._ws)
                if picked is not None:
                    node, adapter_key = picked
                    if node is not None:
                        self._ws._on_node_selected(node, adapter_key)
            if step.tab:
                self._ws._open_tab(step.tab)
            if step.after is not None:
                step.after(self._ws)
        except Exception as exc:  # noqa: BLE001 - a tour must not crash the app
            self._body.setText(step.narration + f"<br><br><i>(could not load the "
                               f"live view: {exc})</i>")

        text = step.callout(self._ws) if callable(step.callout) else step.callout
        self._callout.setVisible(bool(text))
        if text:
            self._callout.setText(f"📊 &nbsp;{text}")
