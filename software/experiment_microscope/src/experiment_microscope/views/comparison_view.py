"""meeting01 ｜ thesis structural comparison (FIXME §22).

This view exists to make the *differences* legible, not to argue the two
pipelines are the same. Both use spiking autoencoders and paraconsistent
engineering; that is a shared vocabulary, not a shared method. Every row is
tagged with one relation:

    same concept              the two cells name the same idea, implemented
                              however each pipeline implements it
    similar implementation    same idea, comparable code path / parameters
    different implementation  same idea, materially different mechanism
    not applicable            the aspect only exists in one pipeline
    unknown                   not established from the artifacts / code

The relation column is colour-coded; nothing here implies numeric equivalence.
Values are grounded in the two experiments' source and their persisted
`summary.json` / event logs (hover a cell for the origin).
"""

from __future__ import annotations

from experiment_microscope.views._help import HelpBox

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

SAME = "same concept"
SIMILAR = "similar implementation"
DIFFERENT = "different implementation"
NA = "not applicable"
UNKNOWN = "unknown"

_RELATION_COLOR = {
    SAME: QColor(70, 130, 90),
    SIMILAR: QColor(90, 120, 160),
    DIFFERENT: QColor(170, 120, 60),
    NA: QColor(90, 90, 90),
    UNKNOWN: QColor(120, 80, 120),
}


_HELP = """
<b>What this shows.</b> A structural, side-by-side comparison of the two
pipelines. It is <b>not</b> a claim that they are scientifically equivalent.
<br><br>
Each row is one aspect (dataset, windowing, encoding, wavelet, the autoencoder
families, cross-validation, …). The relation column is one of:
<b>same concept</b>, <b>similar implementation</b>, <b>different
implementation</b>, <b>not applicable</b>, <b>unknown</b> — colour-coded, with the
source of the claim in the tooltip. The wording is deliberately conservative.
"""


@dataclass(frozen=True)
class Row:
    aspect: str
    meeting01: str
    thesis: str
    relation: str
    source: str = ""


# Curated from software/nn/src/experiments/{meeting01,thesis}/ and the persisted
# results. Kept deliberately conservative — DIFFERENT / UNKNOWN when in doubt.
_ROWS: tuple[Row, ...] = (
    Row("Task", "spoken-digit representation learning (unsupervised AE)",
        "speaker/subject authentication from EEG or voice", DIFFERENT,
        "meeting01 Experiment.cpp / thesis Phase-01 classifiers"),
    Row("Primary dataset", "FSDD (also AudioMNIST, MIT-BIH)",
        "in-house EEG + voice corpus (~/database.sqlite)", DIFFERENT, "adapters"),
    Row("Signal", "1-D audio, z-scored 256-sample windows",
        "6-ch EEG @ 1024 Hz (4096) or voice @ 44.1 kHz", DIFFERENT,
        "Meeting01Dataset.hpp / ThesisDataset.hpp"),
    Row("Windowing", "fixed 256-sample windows, per-window z-score",
        "whole-trial; no sliding window", DIFFERENT, "to_window_tensor"),
    Row("Spike encoding", "direct / poisson / latency (configurable)",
        "handcrafted path has none; DSNN path encodes internally", DIFFERENT,
        "Meeting01Encoding.hpp"),
    Row("Wavelet transform", "not in the main path (optional analysis only)",
        "DTWPT packet decomposition, daubN, perceptual scale grouping", DIFFERENT,
        "ThesisHandcraftedFeatures.cpp"),
    Row("Paraconsistent engineering", "not used",
        "α/β → g1,g2 → d_penalized ranking of feature sets", NA,
        "ThesisParaconsistent.hpp"),
    Row("SNN autoencoder", "ProtocolSpikingAutoencoder (LIF, Sequential enc/dec)",
        "DSNN classifier head (not an autoencoder)", DIFFERENT,
        "ProtocolSpikingAutoencoder.hpp"),
    Row("ANN / LSTM / GRU / Transformer AE", "baseline autoencoders, same latent dim",
        "not part of the thesis pipeline", NA, "models/lstm, models/transformer"),
    Row("Latent representation", "AE bottleneck vector per window",
        "feature vector (handcrafted) or DSNN embedding", SAME,
        "both produce a per-sample vector"),
    Row("Cross-validation", "nested speaker-disjoint LOSO",
        "subject-grouped folds (Phase-01)", SIMILAR,
        "build_split / thesis phase01 metrics.csv"),
    Row("Seeding", "explicit seed in session_begin event",
        "explicit seed in summary.json (default 42)", SAME, "provenance records"),
    Row("Latent dimension", "config `latent_dim` (per run)",
        "= feature count (handcrafted) / config (DSNN)", DIFFERENT, "configs"),
    Row("Reported metrics", "reconstruction MSE/MAE/R², comparative table",
        "EER / AUC over folds", DIFFERENT,
        "*_comparative_metrics.csv / *_metrics.csv"),
    Row("Reproducibility artifact", "events.jsonl + split_manifest.json",
        "summary.json + paraconsistent.csv + metrics.csv", SIMILAR, "results/"),
)


class ComparisonView(QWidget):
    def __init__(self, repo=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.addWidget(HelpBox('Meeting01 vs Thesis', _HELP))
        title = QLabel(
            "Structural comparison — <b>not</b> a claim of equivalence. "
            "Shared terms (SNN, paraconsistent) do not mean shared method."
        )
        title.setWordWrap(True)
        root.addWidget(title)

        self._table = QTableWidget(len(_ROWS), 4)
        self._table.setHorizontalHeaderLabels(["Aspect", "Meeting01", "Thesis", "Relation"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setWordWrap(True)
        for r, row in enumerate(_ROWS):
            for c, text in enumerate((row.aspect, row.meeting01, row.thesis)):
                item = QTableWidgetItem(text)
                if row.source:
                    item.setToolTip(f"source: {row.source}")
                if c == 0:
                    item.setFont(_bold(item))
                self._table.setItem(r, c, item)
            rel = QTableWidgetItem(row.relation)
            rel.setForeground(QColor("white"))
            rel.setBackground(_RELATION_COLOR[row.relation])
            rel.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(r, 3, rel)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.resizeRowsToContents()
        root.addWidget(self._table, 1)

        legend = QLabel(
            "  ".join(f"<span style='background:{_RELATION_COLOR[k].name()};"
                      f"color:white'>&nbsp;{k}&nbsp;</span>"
                      for k in (SAME, SIMILAR, DIFFERENT, NA, UNKNOWN))
        )
        legend.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(legend)

    def show_node(self, node, adapter_key: str) -> None:  # noqa: D401 - uniform view API
        """The comparison is pipeline-wide; a selection does not change it."""

    def relation_counts(self) -> dict[str, int]:
        counts = {k: 0 for k in _RELATION_COLOR}
        for row in _ROWS:
            counts[row.relation] += 1
        return counts


def _bold(item: QTableWidgetItem):
    f = item.font()
    f.setBold(True)
    return f
