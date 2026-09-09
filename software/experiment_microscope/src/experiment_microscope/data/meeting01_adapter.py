"""meeting01-pipeline adapter (FIXME §1, §41, §45).

Two data sources:

* **Event stream / timeline / ranking** — parsed by importing
  ``software/nn/scripts/pipeline/meeting01/monitor.py`` directly (it is
  import-safe, stdlib-only: ``EventTailer``, ``SessionState``, ``_drain``,
  ``render_plain``). Reads ``results/meeting01/meeting01_loso_<ds>_fold<f>_events.jsonl``
  and the per-fold ``*_comparative_metrics.csv`` / ``*_per_window_errors.csv``.
* **Signals / windows / encodings / latents / reconstructions** — recomputed
  live through ``nn_microscope.meeting01``. These work even with zero persisted
  results, which is the state right after the purge / before the weeks-long
  reprocess.

The dataset list (fsdd / audiomnist / mitbih) comes from the LOSO profile so
the explorer is populated before any run exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.data.adapters import (
    ExperimentAdapter,
    ProvenanceRecord,
    TreeNode,
)
from experiment_microscope.paths import (
    MEETING01_MONITOR_DIR,
    MEETING01_PROFILES,
    MEETING01_RESULTS,
)

_RUN_TAG = "meeting01_loso"
_LOSO_PROFILE = "meeting01-loso.json"


def _load_monitor() -> ModuleType:
    """Import monitor.py without perturbing the meeting01 package namespace."""
    mod_path = MEETING01_MONITOR_DIR / "monitor.py"
    if not mod_path.is_file():
        raise FileNotFoundError(f"meeting01 monitor not found at {mod_path}")
    if "meeting01_monitor" in sys.modules:
        return sys.modules["meeting01_monitor"]
    spec = importlib.util.spec_from_file_location("meeting01_monitor", mod_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["meeting01_monitor"] = module
    return module


class Meeting01Adapter(ExperimentAdapter):
    key = "meeting01"
    title = "Meeting01"

    def __init__(self, results_dir: Path | None = None, profile_dir: Path | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else MEETING01_RESULTS
        self.profile_dir = Path(profile_dir) if profile_dir else MEETING01_PROFILES
        self._profile_cache: dict[str, Any] | None = None

    # -- profile --------------------------------------------------
    def _profile(self) -> dict[str, Any]:
        if self._profile_cache is None:
            path = self.profile_dir / _LOSO_PROFILE
            self._profile_cache = json.loads(path.read_text()) if path.is_file() else {}
        return self._profile_cache

    def _datasets(self) -> list[dict[str, Any]]:
        prof = self._profile()
        ds = (prof.get("dataset") or {})
        sources = ds.get("sources") or []
        if sources:
            return sources
        # legacy flat form: single dataset
        return [{"name": prof.get("dataset", {}).get("name", "fsdd")}]

    def _cv_folds(self) -> int:
        ds = self._profile().get("dataset") or {}
        return int(ds.get("cv_num_folds", 6))

    # -- session state (may be empty) -----------------------------
    def session_state(self):
        try:
            monitor = _load_monitor()
        except FileNotFoundError:
            return None
        if not self.results_dir.is_dir():
            return None
        return monitor._drain(str(self.results_dir), _RUN_TAG)

    def dashboard_text(self) -> str | None:
        state = self.session_state()
        if state is None:
            return None
        monitor = _load_monitor()
        return monitor.render_plain(state)

    # -- discovery -----------------------------------------------
    def root_nodes(self) -> list[TreeNode]:
        return [
            TreeNode(
                kind="experiment",
                label=self.title,
                handle={"level": "root"},
                metadata={
                    "results_dir": str(self.results_dir),
                    "has_results": self.results_dir.is_dir(),
                    "profile": str(self.profile_dir / _LOSO_PROFILE),
                },
            )
        ]

    def children(self, node: TreeNode) -> list[TreeNode]:
        h = node.handle
        level = h.get("level")
        if level == "root":
            return [
                TreeNode(
                    "dataset",
                    src.get("name", "?"),
                    {"level": "dataset", "dataset": src.get("name")},
                    metadata=dict(src),
                )
                for src in self._datasets()
            ]
        if level == "dataset":
            n = self._cv_folds()
            return [
                TreeNode(
                    "fold",
                    f"outer fold {f}",
                    {"level": "fold", "dataset": h["dataset"], "cv_fold": f},
                    metadata=self._fold_metadata(h["dataset"], f),
                )
                for f in range(n)
            ]
        if level == "fold":
            # Encodings x architectures are the recompute leaves (FIXME §43-B).
            evaluation = self._profile().get("evaluation") or {}
            encs = evaluation.get("encodings", ["direct", "poisson", "latency"])
            archs = evaluation.get("snn_architectures", ["dense", "conv1d", "recurrent"])
            leaves = []
            for enc in encs:
                for arch in archs:
                    leaves.append(
                        TreeNode(
                            "model",
                            f"snn-ae / {arch} / {enc}",
                            {
                                "level": "combo",
                                "dataset": h["dataset"],
                                "cv_fold": h["cv_fold"],
                                "encoding": enc,
                                "architecture": arch,
                            },
                            has_children=False,
                        )
                    )
            return leaves
        return []

    def _fold_metadata(self, dataset: str, fold: int) -> dict[str, Any]:
        events = self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_events.jsonl"
        metrics = self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_comparative_metrics.csv"
        return {
            "events_jsonl": str(events) if events.is_file() else None,
            "comparative_metrics_csv": str(metrics) if metrics.is_file() else None,
            "status": "completed" if metrics.is_file() else ("running" if events.is_file() else "not started"),
        }

    # -- provenance (from session_begin) -------------------------
    def load_provenance(self, node: TreeNode) -> ProvenanceRecord:
        h = node.handle
        dataset = h.get("dataset")
        fold = h.get("cv_fold")
        begin: dict[str, Any] = {}
        events = self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_events.jsonl"
        if events.is_file():
            with events.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "session_begin":
                        begin = ev
                        break

        def mv(key: str) -> Value:
            return Value(begin[key], Origin.MEASURED) if key in begin else Value.missing()

        source = {
            "experiment": Value("Meeting01 / nested-LOSO", Origin.MEASURED),
            "dataset": Value(dataset, Origin.MEASURED) if dataset else Value.missing(),
            "cv_fold": Value(fold, Origin.MEASURED) if fold is not None else Value.missing(),
        }
        processing = {
            "encoding": Value(h.get("encoding"), Origin.MEASURED) if h.get("encoding") else Value.missing(),
            "architecture": Value(h.get("architecture"), Origin.MEASURED) if h.get("architecture") else Value.missing(),
            "window_size": mv("window_size"),
        }
        model = {
            "seed": mv("seed"),
            "repeats": mv("repeats"),
        }
        artifact = {
            "run_tag": Value(_RUN_TAG, Origin.MEASURED),
            "git_commit": mv("git_commit"),
            "backend": mv("backend"),
            "events_jsonl": Value(str(events), Origin.MEASURED) if events.is_file() else Value.missing(),
        }
        return ProvenanceRecord(source=source, processing=processing, model=model, artifact=artifact)
