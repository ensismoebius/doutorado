"""TransformationCache + worker offload (FIXME §24, §32).

The GUI thread must never block on a dataset load, a wavelet decomposition or
a latent projection. Views ask the cache for a derived representation keyed by
``(adapter, sample_id, stage, params_hash)``; on a miss the work runs on a
``QThreadPool`` and the result is delivered back on the GUI thread via a
signal. Identical requests already in flight are coalesced.

The animation engine relies on this: stepping frames must not recompute the
underlying transform (FIXME §24 "Cache derived representations").
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


def params_hash(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


@dataclass(frozen=True)
class CacheKey:
    adapter: str
    sample_id: str
    stage: str
    phash: str

    @classmethod
    def make(cls, adapter: str, sample_id: str, stage: str, params: dict[str, Any]) -> "CacheKey":
        return cls(adapter, str(sample_id), stage, params_hash(params))


@dataclass
class _Job(QRunnable):
    key: CacheKey
    fn: Callable[[], Any]
    signals: "_JobSignals" = field(default_factory=lambda: _JobSignals())

    def __post_init__(self) -> None:
        super().__init__()
        self.setAutoDelete(True)

    def run(self) -> None:  # executed on a pool thread
        try:
            value = self.fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the GUI intact
            self.signals.failed.emit(self.key, exc, traceback.format_exc())
            return
        self.signals.done.emit(self.key, value)


class _JobSignals(QObject):
    done = Signal(object, object)          # (CacheKey, value)
    failed = Signal(object, object, str)   # (CacheKey, exception, traceback)


class TransformationCache(QObject):
    """LRU cache of derived representations with async fill."""

    #: (CacheKey, value) — a requested representation is ready.
    ready = Signal(object, object)
    #: (CacheKey, exception, traceback_text) — computation raised. The GUI
    #: surfaces the message unchanged (no-fallback policy).
    failed = Signal(object, object, str)

    def __init__(self, max_entries: int = 256, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._store: "OrderedDict[CacheKey, Any]" = OrderedDict()
        self._inflight: set[CacheKey] = set()
        self._max = max_entries
        self._pool = QThreadPool.globalInstance()

    def peek(self, key: CacheKey) -> Any | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def request(self, key: CacheKey, fn: Callable[[], Any]) -> Any | None:
        """Return the cached value now, or schedule ``fn`` and return None.

        When the value later arrives, ``ready`` fires with ``(key, value)``.
        """

        hit = self.peek(key)
        if hit is not None:
            self.ready.emit(key, hit)
            return hit
        if key in self._inflight:
            return None
        self._inflight.add(key)
        job = _Job(key, fn)
        job.signals.done.connect(self._on_done)
        job.signals.failed.connect(self._on_failed)
        self._pool.start(job)
        return None

    def compute_blocking(self, key: CacheKey, fn: Callable[[], Any]) -> Any:
        """Synchronous fill — for tests and headless use."""
        hit = self.peek(key)
        if hit is not None:
            return hit
        value = fn()
        self._insert(key, value)
        return value

    def _insert(self, key: CacheKey, value: Any) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    def _on_done(self, key: CacheKey, value: Any) -> None:
        self._inflight.discard(key)
        self._insert(key, value)
        self.ready.emit(key, value)

    def _on_failed(self, key: CacheKey, exc: BaseException, tb: str) -> None:
        self._inflight.discard(key)
        self.failed.emit(key, exc, tb)

    def clear(self) -> None:
        self._store.clear()
