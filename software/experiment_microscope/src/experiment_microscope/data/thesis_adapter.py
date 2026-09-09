"""Thesis-pipeline adapter (FIXME §2, §41).

On-disk artifacts (``software/nn/results/thesis/``):

* ``phase00/e05_<run_tag>_paraconsistent.csv``  — feature-set ranking (α β G1 G2 D_truth D_penalized)
* ``phase00/e05_<run_tag>_summary.json``        — config, seed, dataset composition, best_*
* ``phase01/e05_<run_tag>_metrics.csv``         — EER / AUC per fold, model_path
* ``phase01/models/<run_tag>/<feature>/fold_*.bin``

Nothing per-sample is persisted. Raw signals, wavelet decompositions, feature
vectors and per-sample paraconsistent inputs are recomputed on demand through
``nn_microscope.thesis`` / ``nn_microscope.wavelet`` from the dataset at
``~/database.sqlite``. Until that binding is built those payloads raise with
the build command (no-fallback).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.data.adapters import (
    ExperimentAdapter,
    FeatureMatrix,
    ParaconsistentPoint,
    ProvenanceRecord,
    TreeNode,
    missing_map,
)
from experiment_microscope.data.adapters import Signal1D
from experiment_microscope.paths import THESIS_DEFAULT_DB, THESIS_RESULTS
from experiment_microscope.processing._binding import BindingUnavailableError, load_binding

import numpy as np

_PARA_SUFFIX = "_paraconsistent.csv"
_SUMMARY_SUFFIX = "_summary.json"


def _run_tag_from(path: Path) -> str:
    name = path.name
    for suffix in (_PARA_SUFFIX, _SUMMARY_SUFFIX, "_metrics.csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.startswith("e05_"):
        name = name[4:]
    return name


def _read_para_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _v(row: dict[str, Any], key: str, origin: Origin = Origin.MEASURED) -> Value:
    raw = row.get(key, "")
    if raw in ("", "nan", "NaN", None):
        return Value.missing()
    try:
        return Value(float(raw), origin)
    except (TypeError, ValueError):
        return Value(raw, origin)


class ThesisAdapter(ExperimentAdapter):
    key = "thesis"
    title = "Thesis"

    def __init__(self, results_dir: Path | None = None, db_path: Path | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else THESIS_RESULTS
        self.db_path = Path(db_path) if db_path else THESIS_DEFAULT_DB
        self._view_cache: dict[str, Any] = {}  # modality -> nn_microscope DatasetView
        self._feat_cache: dict[str, FeatureMatrix] = {}  # run_tag -> recomputed matrix

    # -- live dataset (recomputed via nn_microscope.thesis) ---------------
    def _view(self, modality: str):
        v = self._view_cache.get(modality)
        if v is None:
            nm = load_binding()
            v = nm.thesis.load_dataset(str(self.db_path), modality, 0)
            self._view_cache[modality] = v
        return v

    # -- discovery ------------------------------------------------
    def _phase_dir(self, phase: str) -> Path:
        return self.results_dir / phase

    def _run_tags(self, phase: str) -> list[str]:
        d = self._phase_dir(phase)
        if not d.is_dir():
            return []
        tags = {_run_tag_from(p) for p in d.glob(f"*{_PARA_SUFFIX}")}
        return sorted(tags)

    def root_nodes(self) -> list[TreeNode]:
        present = self.results_dir.is_dir()
        return [
            TreeNode(
                kind="experiment",
                label=self.title,
                handle={"level": "root"},
                metadata={"results_dir": str(self.results_dir), "on_disk": present},
                has_children=present,
            )
        ]

    def children(self, node: TreeNode) -> list[TreeNode]:
        h = node.handle
        level = h.get("level")
        if level == "root":
            out = []
            for phase, label in (("phase00", "Phase 00 — feature ranking"),
                                 ("phase01", "Phase 01 — DSNN authentication")):
                if self._run_tags(phase):
                    out.append(TreeNode("run", label, {"level": "phase", "phase": phase}))
            return out
        if level == "phase":
            phase = h["phase"]
            return [
                TreeNode("run", tag, {"level": "run", "phase": phase, "run_tag": tag},
                         metadata=self._run_metadata(phase, tag))
                for tag in self._run_tags(phase)
            ]
        if level == "run":
            phase, tag = h["phase"], h["run_tag"]
            rows = self._para_rows(phase, tag)
            out: list[TreeNode] = [
                TreeNode(
                    "result",
                    r["label"],
                    {"level": "feature_set", "phase": phase, "run_tag": tag, "feature_set": r["label"]},
                    metadata={k: r[k] for k in r},
                    has_children=False,
                )
                for r in rows
            ]
            modality = (self._run_metadata(phase, tag) or {}).get("modality")
            if modality in ("eeg", "voice", "fused"):
                out.append(
                    TreeNode(
                        "dataset",
                        f"samples ({modality}, live)",
                        {"level": "samples", "phase": phase, "run_tag": tag, "modality": modality},
                    )
                )
            return out
        if level == "samples":
            try:
                view = self._view(h["modality"])
            except BindingUnavailableError:
                return []
            subj = view.subject_ids
            stim = view.stimuli
            n = min(view.n_samples, 200)  # lazy cap (FIXME §7)
            return [
                TreeNode(
                    "sample",
                    f"#{i}  subj {subj[i]}  stim {stim[i]}",
                    {"level": "sample", "modality": h["modality"], "index": i},
                    metadata={"subject_id": subj[i], "stimulus": stim[i]},
                    has_children=False,
                )
                for i in range(n)
            ]
        return []

    def load_features(self, node: TreeNode) -> FeatureMatrix:
        """Recompute a Phase-00 run's handcrafted feature matrix live (FIXME §14).

        Uses the run's own ``summary.json`` handcrafted config so the matrix is
        the one that produced the persisted ranking.
        """
        h = node.handle
        if h.get("level") != "run":
            raise NotImplementedError("select a Phase-00 run node to recompute its feature matrix")
        phase, tag = h.get("phase"), h.get("run_tag")
        if not tag:
            raise NotImplementedError("select a Phase-00 run")
        s = self._summary(phase, tag)
        hc = s.get("handcrafted") or {}
        modality = s.get("modality") or "eeg"
        if s.get("strategy") != "handcrafted":
            raise NotImplementedError("live feature matrix is only wired for handcrafted runs")
        cached = self._feat_cache.get(tag)
        if cached is not None:
            return cached
        nm = load_binding()
        view = self._view(modality)
        sets = nm.thesis.extract_handcrafted_features(
            view,
            modality=modality,
            transform=hc.get("transform", "dtwpt"),
            scale=hc.get("scale", "lfcc"),
            descriptors=list(hc.get("descriptors", ["energy", "zcr", "entropy", "teager"])),
            dtwpt_level=int(hc.get("dtwpt_level", 4)),
            wavelet=hc.get("wavelet", "daub4"),
            cepstral=bool(hc.get("cepstral", False)),
            seed=int(s.get("seed", 42)),
        )
        fs = sets[0]
        values = np.asarray(fs.vectors, dtype=float)
        subj = list(view.subject_ids)
        matrix = FeatureMatrix(
            values=values,
            feature_names=tuple(f"f{j}" for j in range(values.shape[1])),
            sample_labels=tuple(f"#{i} subj{subj[i]}" for i in range(values.shape[0])),
            class_labels=tuple(subj),
            origin=Origin.COMPUTED,
            set_label=f"{tag} / {fs.label}",
        )
        self._feat_cache[tag] = matrix
        return matrix

    def load_signal(self, node: TreeNode) -> Signal1D:
        h = node.handle
        if h.get("level") != "sample":
            raise NotImplementedError("select an individual thesis sample")
        modality = h["modality"]
        view = self._view(modality)
        s = view.sample(h["index"])
        if modality == "voice":
            data = np.asarray(s["audio"]).reshape(-1)
            return Signal1D(
                samples=data, sample_rate=44100.0, origin=Origin.MEASURED,
                unit="amplitude",
                label=f"voice sample #{h['index']} (subj {s['subject_id']}, stim {s['stimulus']})",
            )
        eeg = np.asarray(s["eeg"])  # (channels, samples)
        return Signal1D(
            samples=eeg, sample_rate=1024.0, origin=Origin.MEASURED,
            channel_names=tuple(f"ch{c}" for c in range(eeg.shape[0])), unit="µV",
            label=f"EEG sample #{h['index']} (subj {s['subject_id']}, stim {s['stimulus']})",
        )

    # -- helpers ------------------------------------------------
    def _para_rows(self, phase: str, run_tag: str) -> list[dict[str, Any]]:
        path = self._phase_dir(phase) / f"e05_{run_tag}{_PARA_SUFFIX}"
        return _read_para_csv(path) if path.is_file() else []

    def _summary(self, phase: str, run_tag: str) -> dict[str, Any]:
        path = self._phase_dir(phase) / f"e05_{run_tag}{_SUMMARY_SUFFIX}"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

    def _run_metadata(self, phase: str, run_tag: str) -> dict[str, Any]:
        s = self._summary(phase, run_tag)
        if not s:
            return {"run_tag": run_tag}
        return {
            "run_tag": run_tag,
            "modality": s.get("modality"),
            "strategy": s.get("strategy"),
            "seed": s.get("seed"),
            "text_mode": s.get("text_mode"),
            "n_samples": (s.get("dataset") or {}).get("n_samples"),
            "n_subjects": (s.get("dataset") or {}).get("n_subjects"),
            "config_hash": s.get("config_hash"),
        }

    # -- payloads ------------------------------------------------
    def load_provenance(self, node: TreeNode) -> ProvenanceRecord:
        h = node.handle
        phase, tag = h.get("phase"), h.get("run_tag")
        s = self._summary(phase, tag) if phase and tag else {}
        ds = s.get("dataset") or {}
        ae = s.get("autoencoder") or {}
        tr = s.get("training") or {}
        source = {
            "experiment": Value(f"Thesis / {phase}", Origin.MEASURED),
            "modality": Value(s.get("modality"), Origin.MEASURED) if s.get("modality") else Value.missing(),
            "n_subjects": Value(ds.get("n_subjects"), Origin.MEASURED) if ds.get("n_subjects") is not None else Value.missing(),
            "n_stimuli": Value(ds.get("n_stimuli"), Origin.MEASURED) if ds.get("n_stimuli") is not None else Value.missing(),
            "n_samples": Value(ds.get("n_samples"), Origin.MEASURED) if ds.get("n_samples") is not None else Value.missing(),
        }
        processing = {
            "strategy": Value(s.get("strategy"), Origin.MEASURED) if s.get("strategy") else Value.missing(),
            "encoder_layer_spec": Value(", ".join(ae.get("encoder_layer_spec", [])) or None, Origin.MEASURED)
            if ae.get("encoder_layer_spec") else Value.missing(),
            "text_mode": Value(s.get("text_mode"), Origin.MEASURED) if s.get("text_mode") else Value.missing(),
        }
        model = {
            "model": Value(ae.get("model"), Origin.MEASURED) if ae.get("model") else Value.missing(),
            "seed": Value(s.get("seed"), Origin.MEASURED) if s.get("seed") is not None else Value.missing(),
            "epochs": Value(tr.get("epochs"), Origin.MEASURED) if tr.get("epochs") is not None else Value.missing(),
            "learning_rate": Value(tr.get("learning_rate"), Origin.MEASURED) if tr.get("learning_rate") is not None else Value.missing(),
        }
        artifact = {
            "run_tag": Value(tag, Origin.MEASURED) if tag else Value.missing(),
            "config_hash": Value(s.get("config_hash"), Origin.MEASURED) if s.get("config_hash") is not None else Value.missing(),
            "summary_json": Value(str(self._phase_dir(phase) / f"e05_{tag}{_SUMMARY_SUFFIX}"), Origin.MEASURED)
            if phase and tag else Value.missing(),
        }
        return ProvenanceRecord(source=source, processing=processing, model=model, artifact=artifact)

    def paraconsistent_points(self) -> list[ParaconsistentPoint]:
        points: list[ParaconsistentPoint] = []
        for phase in ("phase00", "phase01"):
            d = self._phase_dir(phase)
            if not d.is_dir():
                continue
            for csv_path in sorted(d.glob(f"*{_PARA_SUFFIX}")):
                run_tag = _run_tag_from(csv_path)
                meta = self._run_metadata(phase, run_tag)
                for row in _read_para_csv(csv_path):
                    points.append(
                        ParaconsistentPoint(
                            label=f"{run_tag} / {row.get('label', '?')}",
                            alpha=_v(row, "alpha"),
                            beta=_v(row, "beta"),
                            g1=_v(row, "g1"),
                            g2=_v(row, "g2"),
                            d_truth=_v(row, "d_truth"),
                            d_penalized=_v(row, "d_penalized"),
                            facet={
                                "phase": phase,
                                "run_tag": run_tag,
                                "feature_set": row.get("label"),
                                "modality": meta.get("modality"),
                                "strategy": meta.get("strategy"),
                                "seed": meta.get("seed"),
                            },
                        )
                    )
        return points

    def load_metrics(self, node: TreeNode) -> dict[str, Value]:
        h = node.handle
        phase, tag = h.get("phase"), h.get("run_tag")
        if phase != "phase01" or not tag:
            return missing_map("eer", "auc")
        path = self._phase_dir(phase) / f"e05_{tag}_metrics.csv"
        if not path.is_file():
            return missing_map("eer", "auc")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        want = h.get("feature_set")
        rows = [r for r in rows if not want or r.get("feature_set") == want]
        if not rows:
            return missing_map("eer", "auc")

        def _mean(key: str) -> Value:
            vals = []
            for r in rows:
                try:
                    f = float(r[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if f == f:  # not NaN
                    vals.append(f)
            if not vals:
                return Value.missing()
            return Value(sum(vals) / len(vals), Origin.MEASURED, note=f"mean over {len(vals)} fold(s)")

        return {"eer": _mean("eer"), "auc": _mean("auc"),
                "train_ms": _mean("train_ms"), "infer_ms": _mean("infer_ms")}
