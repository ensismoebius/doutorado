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

import numpy as np

import re

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.data.adapters import (
    ExperimentAdapter,
    LatentTrace,
    ProvenanceRecord,
    Signal1D,
    TreeNode,
)
from experiment_microscope.processing._binding import BindingUnavailableError, load_binding
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
    # Register before exec: monitor.py defines @dataclass classes, and on
    # Python 3.14 dataclasses resolves ``sys.modules.get(cls.__module__)``
    # while processing the class body — which is None if we wait.
    sys.modules["meeting01_monitor"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("meeting01_monitor", None)
        raise
    return module


def _recon_metrics(original: np.ndarray, recon: np.ndarray) -> dict[str, Value]:
    """MSE / MAE / R2 / Pearson r between a signal and its reconstruction (FIXME §20)."""
    a = np.asarray(original, dtype=float).reshape(-1)
    b = np.asarray(recon, dtype=float).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return {k: Value.missing() for k in ("mse", "mae", "r2", "pearson_r")}
    a, b = a[:n], b[:n]
    resid = a - b
    mse = float(np.mean(resid**2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    if a.std() > 0 and b.std() > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    else:
        r = float("nan")
    return {
        "mse": Value(mse, Origin.COMPUTED),
        "mae": Value(float(np.mean(np.abs(resid))), Origin.COMPUTED),
        "r2": Value(r2, Origin.COMPUTED) if r2 == r2 else Value.missing(),
        "pearson_r": Value(r, Origin.COMPUTED) if r == r else Value.missing(),
    }


class Meeting01Adapter(ExperimentAdapter):
    key = "meeting01"
    title = "Meeting01"

    def __init__(self, results_dir: Path | None = None, profile_dir: Path | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else MEETING01_RESULTS
        self.profile_dir = Path(profile_dir) if profile_dir else MEETING01_PROFILES
        self._profile_cache: dict[str, Any] | None = None
        self._split_cache: dict[tuple[str, int], dict[str, Any]] = {}

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
            ds, fold = h["dataset"], h["cv_fold"]
            evaluation = self._profile().get("evaluation") or {}
            encs = evaluation.get("encodings", ["direct", "poisson", "latency"])
            archs = evaluation.get("snn_architectures", ["dense", "conv1d", "recurrent"])
            out: list[TreeNode] = []
            for split in ("test", "val", "train"):
                out.append(
                    TreeNode(
                        "dataset",
                        f"{split} windows (held-out speaker)" if split == "test" else f"{split} windows",
                        {"level": "windows", "dataset": ds, "cv_fold": fold, "split": split},
                    )
                )
            # SNN-AE recompute leaves (FIXME §43-B).
            for enc in encs:
                for arch in archs:
                    out.append(
                        TreeNode(
                            "model",
                            f"snn-ae / {arch} / {enc}",
                            {"level": "combo", "dataset": ds, "cv_fold": fold,
                             "encoding": enc, "architecture": arch},
                            has_children=False,
                        )
                    )
            return out
        if level == "windows":
            try:
                split = self._split(h["dataset"], h["cv_fold"])
            except (BindingUnavailableError, RuntimeError):
                return []
            metas = split[f"{h['split']}_meta"]
            n = min(len(metas), 200)  # lazy cap; FIXME §7 "do not load enormous datasets"
            return [
                TreeNode(
                    "sample",
                    f"win {m['window_id']}  spk {m['speaker']}  rec {m['recording_id']}  digit {m['digit']}",
                    {"level": "window", "dataset": h["dataset"], "cv_fold": h["cv_fold"],
                     "split": h["split"], "row": i},
                    metadata=dict(m),
                    has_children=False,
                )
                for i, m in enumerate(metas[:n])
            ]
        return []

    # -- live split cache (build_split ~1s; reused across views) ----------
    def _split(self, dataset: str, cv_fold: int) -> dict[str, Any]:
        key = (dataset, cv_fold)
        cached = self._split_cache.get(key)
        if cached is None:
            nm = load_binding()
            profile = str(self.profile_dir / _LOSO_PROFILE)
            cached = nm.meeting01.build_split(profile, dataset, cv_fold)
            self._split_cache[key] = cached
        return cached

    def load_signal(self, node: TreeNode) -> Signal1D:
        h = node.handle
        if h.get("level") != "window":
            raise NotImplementedError("select an individual window")
        split = self._split(h["dataset"], h["cv_fold"])
        samples = split[f"{h['split']}_meta"]
        window = split[f"{h['split']}_samples"][h["row"]]
        meta = samples[h["row"]]
        sr = float((self._datasets_by_name().get(h["dataset"]) or {}).get("sample_rate") or 0.0)
        return Signal1D(
            samples=np.asarray(window).reshape(-1),
            sample_rate=sr,
            origin=Origin.COMPUTED,  # windowed + z-scored by nn_microscope, matching the experiment
            unit="z-score",
            label=f"{h['dataset']} {h['split']} window {meta['window_id']} "
            f"(spk {meta['speaker']}, digit {meta['digit']})",
        )

    def _datasets_by_name(self) -> dict[str, dict[str, Any]]:
        return {d.get("name"): d for d in self._datasets()}

    def _fold_metadata(self, dataset: str, fold: int) -> dict[str, Any]:
        events = self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_events.jsonl"
        metrics = self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_comparative_metrics.csv"
        return {
            "events_jsonl": str(events) if events.is_file() else None,
            "comparative_metrics_csv": str(metrics) if metrics.is_file() else None,
            "status": "completed" if metrics.is_file() else ("running" if events.is_file() else "not started"),
        }

    # -- provenance (from session_begin) -------------------------
    # -- trained SNN-AE models (Step E .npz artifacts) -----------
    #: e.g. ...snn_combo_fsdd_direct_dense_vth0_500000_a0_800000_fold0_run1_encoder.npz
    #: — vth / a are floats with the '.' sanitized to '_' in the filename.
    _NPZ_RE = re.compile(
        r"_snn_(?P<role>[a-z]+)_(?P<dataset>[a-z0-9]+)_(?P<encoding>[a-z]+)_(?P<arch>[a-z0-9]+)_"
        r"vth(?P<vth>\d+_\d+)_a(?P<alpha>\d+_\d+)_fold(?P<fold>\d+)_run(?P<run>\d+)_encoder\.npz$"
    )

    @staticmethod
    def _sanitized_float(s: str) -> float:
        return float(s.replace("_", ".", 1))

    def _snn_model_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for enc_npz in sorted(self.results_dir.glob("models/**/*_snn_*_encoder.npz")):
            m = self._NPZ_RE.search(enc_npz.name)
            dec_npz = enc_npz.with_name(enc_npz.name[: -len("_encoder.npz")] + "_decoder.npz")
            if not m or not dec_npz.is_file():
                continue
            specs.append({
                "encoder": str(enc_npz), "decoder": str(dec_npz), "role": m["role"],
                "dataset": m["dataset"], "encoding": m["encoding"], "architecture": m["arch"],
                "v_th": self._sanitized_float(m["vth"]), "alpha": self._sanitized_float(m["alpha"]),
                "fold": int(m["fold"]), "run": int(m["run"]),
            })
        # prefer the retrained winner ("final") over the grid ("combo")
        specs.sort(key=lambda s: (s["role"] != "final", s["architecture"], s["run"]))
        return specs

    def load_latent(self, node: TreeNode, **params: Any) -> LatentTrace:
        h = getattr(node, "handle", {}) or {}
        if h.get("level") != "window":
            raise NotImplementedError("select an individual window")
        specs = self._snn_model_specs()
        if not specs:
            raise RuntimeError(
                "no trained SNN-AE .npz under "
                f"{self.results_dir / 'models'} — run a LOSO fold with "
                "`dataset.save_models: true` (emits *_encoder.npz / *_decoder.npz via Step E) "
                "then reselect this window."
            )
        ds, fold = h.get("dataset"), h.get("cv_fold")
        want_enc, want_arch = h.get("encoding") or params.get("encoding"), h.get("architecture")
        cand = [s for s in specs if s["dataset"] == ds and s["fold"] == fold] or specs
        if want_enc:
            cand = [s for s in cand if s["encoding"] == want_enc] or cand
        if want_arch:
            cand = [s for s in cand if s["architecture"] == want_arch] or cand

        from experiment_microscope.processing import meeting01 as m01

        split = self._split(h["dataset"], h["cv_fold"])
        window = np.asarray(split[f"{h['split']}_samples"][h["row"]], dtype=float)
        # try candidates in order — a model still being written by a live LOSO
        # run, or one whose saved shape does not match make_snn_cfg, fails to
        # load; fall through to the next rather than blanking the view.
        trace = None
        last_err: Exception | None = None
        for spec in cand:
            try:
                trace = m01.snn_ae_forward(
                    str(self.profile_dir / _LOSO_PROFILE),
                    alpha=spec["alpha"], v_th=spec["v_th"], architecture=spec["architecture"],
                    encoder_npz=spec["encoder"], decoder_npz=spec["decoder"],
                    flat_window=window, encoding=spec["encoding"],
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if trace is None:
            raise RuntimeError(
                f"found {len(cand)} SNN-AE .npz for this fold but none loaded "
                f"(a LOSO run may still be writing them): {last_err}"
            )
        original = np.asarray(trace.encoded_input).reshape(-1)
        recon = np.asarray(trace.reconstruction).reshape(-1)
        return LatentTrace(
            latent=np.asarray(trace.latent).reshape(-1),
            reconstruction=recon,
            original=original,
            spikes=None,
            v_mem=None,
            origin=Origin.COMPUTED,
            metrics=_recon_metrics(original, recon),
        )

    def artifact_files(self, node: TreeNode) -> list[str]:
        h = getattr(node, "handle", {}) or {}
        dataset, fold = h.get("dataset"), h.get("cv_fold")
        if dataset is None or fold is None:
            return []
        cands = [
            self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_events.jsonl",
            self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_comparative_metrics.csv",
            self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_per_window_errors.csv",
            self.results_dir / f"{_RUN_TAG}_{dataset}_fold{fold}_split_manifest.json",
        ]
        return [str(p) for p in cands if p.is_file()]

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
