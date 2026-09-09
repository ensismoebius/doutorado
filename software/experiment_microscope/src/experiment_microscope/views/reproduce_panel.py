"""Experiment reproduction panel (FIXME §31).

For the selected run: its configuration, its result files, and the exact
command that regenerates it — as *copyable text*, never executed. Opening the
GUI must never start an experiment (§31); this panel only tells the researcher
what to type.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _thesis_cmd(phase: str, summary: dict) -> str:
    ph = "phase00" if phase == "phase00" else "phase01"
    seed = summary.get("seed", "?")
    modality = summary.get("modality", "?")
    chash = summary.get("config_hash", "?")
    return (
        f"cd software/nn\n"
        f"# regenerates ALL {ph} profiles (individual runs are not separately\n"
        f"# reproducible from here); this run: modality={modality} seed={seed}\n"
        f"# config_hash={chash} strategy={summary.get('strategy', '?')}\n"
        f"./scripts/testing/run_thesis_profiles.sh {ph}"
    )


def _meeting01_cmd(dataset, fold) -> str:
    return (
        f"cd software/nn\n"
        f"EXPERIMENT_CONFIRMED=1 MEETING01_BUILD=max-performance \\\n"
        f"  ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh "
        f"--dataset {dataset} --cv-fold {fold}\n"
        f"# COST: weeks across all datasets x folds; this is one (dataset, fold) process."
    )


def _ga_cmd() -> str:
    return (
        "cd software/nn\n"
        "EXPERIMENT_CONFIRMED=1 \\\n"
        "  ./scripts/pipeline/paraconsistentGA/01_paraconsistentGA_run_all_profiles.sh"
    )


class ReproducePanel(QWidget):
    def __init__(self, repo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._files: list[str] = []
        self._recipe = ""

        root = QVBoxLayout(self)
        self._title = QLabel("Select a run.")
        self._title.setWordWrap(True)
        root.addWidget(self._title)

        root.addWidget(QLabel("<b>Configuration</b>"))
        self._config = QPlainTextEdit(readOnly=True)
        self._config.setMaximumHeight(160)
        root.addWidget(self._config)

        row = QHBoxLayout()
        self._files_btn = QPushButton("Copy result-file paths")
        self._files_btn.clicked.connect(self._copy_files)
        self._files_btn.setEnabled(False)
        row.addWidget(self._files_btn)
        row.addStretch(1)
        root.addLayout(row)
        self._files_label = QLabel("")
        self._files_label.setWordWrap(True)
        root.addWidget(self._files_label)

        root.addWidget(QLabel("<b>Reproduction command</b> — not executed by this app (§31)"))
        self._cmd = QPlainTextEdit(readOnly=True)
        self._cmd.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        root.addWidget(self._cmd, 1)
        self._copy_btn = QPushButton("Copy command")
        self._copy_btn.clicked.connect(self._copy_cmd)
        self._copy_btn.setEnabled(False)
        root.addWidget(self._copy_btn)

    # -- API --------------------------------------------------
    def show_node(self, node, adapter_key: str) -> None:
        h = getattr(node, "handle", {}) or {}
        adapter = self.repo.adapter(adapter_key)
        try:
            self._files = adapter.artifact_files(node)
        except Exception:  # noqa: BLE001
            self._files = []
        self._files_btn.setEnabled(bool(self._files))
        self._files_label.setText("\n".join(self._files) or "(no persisted result files)")

        cfg_text, recipe, title = "", "", ""
        if adapter_key == "thesis" and h.get("phase") and h.get("run_tag"):
            phase, tag = h["phase"], h["run_tag"]
            summary = {}
            for f in self._files:
                if f.endswith("_summary.json"):
                    try:
                        summary = json.loads(Path(f).read_text())
                    except Exception:  # noqa: BLE001
                        summary = {}
            title = f"thesis · {phase} · {tag}"
            cfg_text = json.dumps(summary, indent=2, default=str) if summary else "(no summary.json)"
            recipe = _thesis_cmd(phase, summary)
        elif adapter_key == "meeting01" and h.get("dataset") is not None and h.get("cv_fold") is not None:
            title = f"meeting01 · {h['dataset']} · fold {h['cv_fold']}"
            try:
                prov = adapter.load_provenance(node)
                cfg_text = "\n".join(
                    f"{k}: {v.display()}" for sect in (prov.source, prov.processing, prov.model)
                    for k, v in sect.items()
                )
            except Exception as exc:  # noqa: BLE001
                cfg_text = f"(provenance unavailable: {exc})"
            recipe = _meeting01_cmd(h["dataset"], h["cv_fold"])
        elif adapter_key == "paraconsistent_ga" and h.get("run_tag"):
            title = f"paraconsistent_ga · {h['run_tag']}"
            cfg_text = "\n".join(self._files) or "(pareto.json / individuals.csv)"
            recipe = _ga_cmd()
        else:
            self._title.setText("Select a run / fold node to see its reproduction recipe.")
            self._config.setPlainText("")
            self._cmd.setPlainText("")
            self._copy_btn.setEnabled(False)
            return

        self._title.setText(title)
        self._config.setPlainText(cfg_text)
        self._recipe = recipe
        self._cmd.setPlainText(recipe)
        self._copy_btn.setEnabled(bool(recipe))

    # -- clipboard -------------------------------------------
    def _copy_cmd(self) -> None:
        QApplication.clipboard().setText(self._recipe)
        self._copy_btn.setText("Copied ✓")

    def _copy_files(self) -> None:
        QApplication.clipboard().setText("\n".join(self._files))
        self._files_btn.setText("Copied ✓")
