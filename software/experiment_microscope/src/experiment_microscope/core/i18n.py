"""Very small runtime translation layer.

Not the Qt ``.ts``/``.qm`` pipeline — the app's user-facing text is plain Python
strings, and the didactic surface (guided tour, glossary, verdicts, help) is
re-rendered on demand, so a dict lookup keyed by the English source string is
enough and needs no compile step.

Usage::

    from experiment_microscope.core.i18n import t
    label = t("Start guided tour")
    msg = t("{n} spikes out of {steps} time steps", n=n, steps=steps)

``t`` returns the translation for the active language, or the (formatted)
English source when there is no entry — so a missing translation degrades to
English, never to a crash or a blank.
"""

from __future__ import annotations

from typing import Callable

LANGUAGES = {"en": "English", "pt_BR": "Português (Brasil)"}

_lang = "en"
_catalogs: dict[str, dict[str, str]] = {}
_listeners: list[Callable[[str], None]] = []


def _load(code: str) -> dict[str, str]:
    if code in _catalogs:
        return _catalogs[code]
    cat: dict[str, str] = {}
    if code == "pt_BR":
        try:
            from experiment_microscope.core.i18n_pt_br import CATALOG

            cat = CATALOG
        except Exception:  # noqa: BLE001 - missing catalog just means English
            cat = {}
    _catalogs[code] = cat
    return cat


def set_language(code: str) -> None:
    global _lang
    if code not in LANGUAGES:
        code = "en"
    _lang = code
    _load(code)
    for fn in list(_listeners):
        try:
            fn(code)
        except Exception:  # noqa: BLE001
            pass


def language() -> str:
    return _lang


def on_language_changed(fn: Callable[[str], None]) -> None:
    _listeners.append(fn)


def t(text: str, /, **kw) -> str:
    """Translate ``text`` for the active language; format with ``kw`` if given."""
    s = text
    if _lang != "en":
        s = _load(_lang).get(text, text)
    if kw:
        try:
            return s.format(**kw)
        except (KeyError, IndexError):
            return text.format(**kw) if kw else text
    return s
