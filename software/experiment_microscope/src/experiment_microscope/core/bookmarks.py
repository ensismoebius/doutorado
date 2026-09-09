"""Bookmarks (FIXME §36).

A bookmark stores enough of the workspace state to return to an interesting
place: the full ``SelectionState`` snapshot plus the active view tab and the
window's dock layout. Persisted as JSON in ``QSettings`` so they survive across
sessions.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from PySide6.QtCore import QSettings

_KEY = "bookmarks/v1"


@dataclass
class Bookmark:
    name: str
    selection: dict[str, Any]
    nav_path: list[str] = field(default_factory=list)  # explorer labels, root first
    active_tab: str = ""
    window_state_b64: str = ""  # QMainWindow.saveState() bytes, base64
    created: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Bookmark":
        return cls(
            name=d.get("name", "unnamed"),
            selection=d.get("selection", {}),
            nav_path=list(d.get("nav_path", [])),
            active_tab=d.get("active_tab", ""),
            window_state_b64=d.get("window_state_b64", ""),
            created=d.get("created", 0.0),
        )

    def window_state(self) -> bytes:
        try:
            return base64.b64decode(self.window_state_b64) if self.window_state_b64 else b""
        except (ValueError, TypeError):
            return b""


class BookmarkStore:
    def __init__(self, org: str, app: str) -> None:
        self._settings = QSettings(org, app)

    def load(self) -> list[Bookmark]:
        raw = self._settings.value(_KEY, "")
        if not raw:
            return []
        try:
            return [Bookmark.from_json(d) for d in json.loads(raw)]
        except (json.JSONDecodeError, TypeError):
            return []

    def save(self, bookmarks: list[Bookmark]) -> None:
        self._settings.setValue(_KEY, json.dumps([b.to_json() for b in bookmarks]))

    def add(self, bookmark: Bookmark) -> list[Bookmark]:
        bookmarks = [b for b in self.load() if b.name != bookmark.name]
        bookmarks.append(bookmark)
        self.save(bookmarks)
        return bookmarks

    def remove(self, name: str) -> list[Bookmark]:
        bookmarks = [b for b in self.load() if b.name != name]
        self.save(bookmarks)
        return bookmarks


def make_bookmark(name: str, selection_snapshot: dict[str, Any], nav_path: list[str],
                  active_tab: str, window_state: bytes) -> Bookmark:
    return Bookmark(
        name=name,
        selection=selection_snapshot,
        nav_path=list(nav_path),
        active_tab=active_tab,
        window_state_b64=base64.b64encode(bytes(window_state)).decode("ascii"),
    )
