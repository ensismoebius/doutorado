"""paraconsistentGA adapter (FIXME §48).

Reads ``software/nn/results/paraconsistentGA/pga_<tag>_pareto.json`` and
``pga_<tag>_individuals.csv``. Estimated latency is UNCALIBRATED — every
latency value is tagged ``ESTIMATED`` and carries that note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.data.adapters import (
    ExperimentAdapter,
    ParaconsistentPoint,
    TreeNode,
)
from experiment_microscope.paths import PARACONSISTENT_GA_RESULTS

_PARETO_SUFFIX = "_pareto.json"


class ParaconsistentGaAdapter(ExperimentAdapter):
    key = "paraconsistent_ga"
    title = "Paraconsistent GA"

    def __init__(self, results_dir: Path | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else PARACONSISTENT_GA_RESULTS

    def _run_tags(self) -> list[str]:
        if not self.results_dir.is_dir():
            return []
        return sorted(
            p.name[: -len(_PARETO_SUFFIX)]
            for p in self.results_dir.glob(f"*{_PARETO_SUFFIX}")
        )

    def _pareto(self, run_tag: str) -> dict[str, Any]:
        path = self.results_dir / f"{run_tag}{_PARETO_SUFFIX}"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

    def root_nodes(self) -> list[TreeNode]:
        present = bool(self._run_tags())
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
        if h.get("level") == "root":
            out = []
            for tag in self._run_tags():
                p = self._pareto(tag)
                out.append(
                    TreeNode(
                        "run",
                        tag,
                        {"level": "run", "run_tag": tag},
                        metadata={
                            "pareto_front_size": p.get("pareto_front_size"),
                            "n_evaluated": p.get("n_evaluated"),
                            "warnings": p.get("warnings"),
                        },
                        has_children=True,
                    )
                )
            return out
        if h.get("level") == "run":
            p = self._pareto(h["run_tag"])
            front = p.get("pareto_front") or []
            return [
                TreeNode(
                    "result",
                    f"rank {ind.get('rank', i)} — d_pen {ind.get('d_penalized_mean', float('nan')):.4f}",
                    {"level": "individual", "run_tag": h["run_tag"], "index": i},
                    metadata=dict(ind),
                    has_children=False,
                )
                for i, ind in enumerate(front)
            ]
        return []

    def paraconsistent_points(self) -> list[ParaconsistentPoint]:
        points: list[ParaconsistentPoint] = []
        for tag in self._run_tags():
            p = self._pareto(tag)
            for i, ind in enumerate(p.get("pareto_front") or []):
                genome = ind.get("genome") or {}
                points.append(
                    ParaconsistentPoint(
                        label=f"{tag} / rank {ind.get('rank', i)}",
                        alpha=Value(ind.get("alpha"), Origin.MEASURED) if "alpha" in ind else Value.missing(),
                        beta=Value(ind.get("beta"), Origin.MEASURED) if "beta" in ind else Value.missing(),
                        g1=Value(ind.get("g1"), Origin.MEASURED) if "g1" in ind else Value.missing(),
                        g2=Value(ind.get("g2"), Origin.MEASURED) if "g2" in ind else Value.missing(),
                        d_truth=Value(ind.get("d_truth"), Origin.MEASURED) if "d_truth" in ind else Value.missing(),
                        d_penalized=Value(ind.get("d_penalized_mean"), Origin.MEASURED, note="mean over seeds")
                        if "d_penalized_mean" in ind else Value.missing(),
                        facet={
                            "run_tag": tag,
                            "feasible": ind.get("feasible"),
                            "est_latency_ms": Value(
                                ind.get("est_latency_ms"), Origin.ESTIMATED, unit="ms",
                                note="UNCALIBRATED (paraconsistentGA)",
                            ),
                            "encoding": genome.get("encoding"),
                            "latent": genome.get("latent"),
                            "depth": genome.get("depth"),
                        },
                    )
                )
        return points
