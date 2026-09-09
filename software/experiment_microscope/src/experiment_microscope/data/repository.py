"""The one object every view goes through to reach data (FIXME §32).

Owns the adapters and the ``TransformationCache``. Views never instantiate an
adapter or touch the filesystem directly.
"""

from __future__ import annotations

from experiment_microscope.core.cache import TransformationCache
from experiment_microscope.data.adapters import ExperimentAdapter, ParaconsistentPoint, TreeNode
from experiment_microscope.data.meeting01_adapter import Meeting01Adapter
from experiment_microscope.data.paraconsistent_ga_adapter import ParaconsistentGaAdapter
from experiment_microscope.data.thesis_adapter import ThesisAdapter


class DataRepository:
    def __init__(self) -> None:
        self.cache = TransformationCache()
        self._adapters: dict[str, ExperimentAdapter] = {
            a.key: a
            for a in (Meeting01Adapter(), ThesisAdapter(), ParaconsistentGaAdapter())
        }

    @property
    def adapters(self) -> tuple[ExperimentAdapter, ...]:
        return tuple(self._adapters.values())

    def adapter(self, key: str) -> ExperimentAdapter:
        return self._adapters[key]

    def adapter_for_node(self, node: TreeNode) -> ExperimentAdapter:
        # Every node's handle originated from an adapter; the explorer stashes
        # the owning key alongside it.
        key = getattr(node, "_adapter_key", None) or node.metadata.get("_adapter_key")
        if key:
            return self._adapters[key]
        raise KeyError("TreeNode has no adapter key")

    def root_nodes(self) -> list[tuple[str, list[TreeNode]]]:
        return [(a.key, a.root_nodes()) for a in self.adapters]

    def all_paraconsistent_points(self) -> list[ParaconsistentPoint]:
        points: list[ParaconsistentPoint] = []
        for a in self.adapters:
            try:
                points.extend(a.paraconsistent_points())
            except Exception:  # noqa: BLE001 - one bad adapter must not blank the plane
                continue
        return points
