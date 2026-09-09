"""Locate and load the compiled ``nn_microscope`` pybind11 module.

FIXME §3 / §198: the GUI must not contain a second implementation of the
scientific pipeline. Any interactive recomputation goes through this module,
which links the actual ``meeting01`` / ``thesis`` C++ libraries. If it is not
built, we raise with the exact build command — never a NumPy stand-in
(no-fallback policy).
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from experiment_microscope.paths import NN_BINDING_BUILD_HINT, NN_BINDING_SEARCH_DIRS

_MODULE_NAME = "nn_microscope"
_cached: ModuleType | None = None


class BindingUnavailableError(RuntimeError):
    """The ``nn_microscope`` extension is not importable on this machine."""


def _find_so() -> str | None:
    for directory in NN_BINDING_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for entry in sorted(directory.glob(f"{_MODULE_NAME}*.so")):
            return str(entry)
    return None


def load_binding() -> ModuleType:
    """Return the imported ``nn_microscope`` module, or raise.

    Result is cached for the process. The failure message names both what is
    missing and how to produce it.
    """

    global _cached
    if _cached is not None:
        return _cached

    if _MODULE_NAME in sys.modules:
        _cached = sys.modules[_MODULE_NAME]
        return _cached

    so_path = _find_so()
    if so_path is None:
        searched = "\n  ".join(str(d) for d in NN_BINDING_SEARCH_DIRS)
        raise BindingUnavailableError(
            f"Compiled extension '{_MODULE_NAME}' not found. Searched:\n  {searched}\n"
            f"Build it with:\n  {NN_BINDING_BUILD_HINT}"
        )

    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, so_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build import spec for {so_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except BaseException as exc:  # noqa: BLE001
        raise BindingUnavailableError(
            f"Found '{so_path}' but importing it failed: {exc!r}. "
            f"Rebuild with:\n  {NN_BINDING_BUILD_HINT}"
        ) from exc

    sys.modules[_MODULE_NAME] = module
    _cached = module
    return module


def is_available() -> bool:
    try:
        load_binding()
        return True
    except BindingUnavailableError:
        return False
