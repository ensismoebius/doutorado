"""Raw artifact inspector (FIXME §28).

When the selected object comes from a file, show the file itself: path, format,
size, and — for arrays / matrices — shape, dtype, min / max / mean / std and the
NaN / Inf counts. The point is that a researcher never has to guess what is
actually on disk behind a plot.

Recompute-only objects (a live thesis feature matrix, a meeting01 window) have
no artifact; the panel says so rather than inventing one.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

_MAX_TEXT = 20_000  # characters of raw content to echo


class ArtifactInspector(QWidget):
    def __init__(self, repo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._files: list[str] = []

        root = QVBoxLayout(self)
        self._picker = QComboBox()
        self._picker.currentIndexChanged.connect(self._render)
        root.addWidget(self._picker)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = self._text.font()
        font.setFamily("monospace")
        self._text.setFont(font)
        root.addWidget(self._text, 1)

    def show_node(self, node, adapter_key: str) -> None:
        try:
            self._files = self.repo.adapter(adapter_key).artifact_files(node)
        except Exception as exc:  # noqa: BLE001
            self._files = []
            self._text.setPlainText(f"artifact_files failed: {exc}")
        self._picker.blockSignals(True)
        self._picker.clear()
        self._picker.addItems([Path(f).name for f in self._files])
        self._picker.blockSignals(False)
        if self._files:
            self._render()
        else:
            self._text.setPlainText(
                "No persisted artifact for this object — it is recomputed live "
                "(nothing to inspect on disk)."
            )

    def _render(self) -> None:
        i = self._picker.currentIndex()
        if not (0 <= i < len(self._files)):
            return
        path = Path(self._files[i])
        self._text.setPlainText(_report(path))


# -- format-specific reports ------------------------------------
def _report(path: Path) -> str:
    if not path.is_file():
        return f"missing: {path}"
    st = path.stat()
    head = [
        f"File   : {path.name}",
        f"Path   : {path}",
        f"Format : {path.suffix.lstrip('.') or 'unknown'}",
        f"Size   : {_human(st.st_size)} ({st.st_size} bytes)",
        "",
    ]
    try:
        if path.suffix == ".npy":
            body = _npy_report(path)
        elif path.suffix == ".npz":
            body = _npz_report(path)
        elif path.suffix == ".csv":
            body = _csv_report(path)
        elif path.suffix == ".json":
            body = _json_report(path)
        elif path.suffix == ".jsonl":
            body = _jsonl_report(path)
        else:
            body = _text_head(path)
    except Exception as exc:  # noqa: BLE001
        body = f"could not parse: {exc}"
    return "\n".join(head) + body


def _array_stats(name: str, a: np.ndarray) -> str:
    lines = [f"  {name}", f"    shape = {a.shape}", f"    dtype = {a.dtype}"]
    if a.size and np.issubdtype(a.dtype, np.number):
        finite = a[np.isfinite(a)]
        lines += [
            f"    min   = {finite.min() if finite.size else float('nan'):.6g}",
            f"    max   = {finite.max() if finite.size else float('nan'):.6g}",
            f"    mean  = {finite.mean() if finite.size else float('nan'):.6g}",
            f"    std   = {finite.std() if finite.size else float('nan'):.6g}",
            f"    NaN   = {int(np.isnan(a).sum())}",
            f"    Inf   = {int(np.isinf(a).sum())}",
        ]
        if a.ndim == 2:
            lines.append(f"    matrix = {a.shape[0]} rows x {a.shape[1]} cols "
                         f"({100.0 * np.count_nonzero(a == 0) / a.size:.1f}% zeros)")
    return "\n".join(lines)


def _npy_report(path: Path) -> str:
    return "Array\n" + _array_stats(path.stem, np.load(path, allow_pickle=False))


def _npz_report(path: Path) -> str:
    with np.load(path, allow_pickle=False) as z:
        return f"Archive: {len(z.files)} array(s)\n" + "\n".join(
            _array_stats(k, z[k]) for k in z.files
        )


def _csv_report(path: Path) -> str:
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "empty CSV"
    header, data = rows[0], rows[1:]
    out = [f"Matrix : {len(data)} rows x {len(header)} columns",
           f"Columns: {', '.join(header)}", ""]
    for c, name in enumerate(header):
        col = []
        for r in data:
            if c < len(r):
                try:
                    col.append(float(r[c]))
                except ValueError:
                    pass
        if col:
            arr = np.array(col)
            nan = int(np.isnan(arr).sum())
            out.append(f"  {name}: min={np.nanmin(arr):.6g} max={np.nanmax(arr):.6g} "
                       f"mean={np.nanmean(arr):.6g} NaN={nan} (n={len(col)})")
    out += ["", "first rows:"]
    for r in data[:15]:
        out.append("  " + " | ".join(r))
    return "\n".join(out)


def _json_report(path: Path) -> str:
    obj = json.loads(path.read_text())
    keys = list(obj.keys()) if isinstance(obj, dict) else f"(top-level {type(obj).__name__})"
    pretty = json.dumps(obj, indent=2, default=str)
    return f"Top-level keys: {keys}\n\n{pretty[:_MAX_TEXT]}"


def _jsonl_report(path: Path) -> str:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    out = [f"Records: {len(lines)}"]
    for label, ln in (("first", lines[0] if lines else None),
                      ("last", lines[-1] if lines else None)):
        if ln:
            try:
                out.append(f"\n{label}:\n{json.dumps(json.loads(ln), indent=2)[:4000]}")
            except json.JSONDecodeError:
                out.append(f"\n{label}: {ln[:2000]}")
    return "\n".join(out)


def _text_head(path: Path) -> str:
    return "raw (head):\n" + path.read_text(errors="replace")[:_MAX_TEXT]


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
