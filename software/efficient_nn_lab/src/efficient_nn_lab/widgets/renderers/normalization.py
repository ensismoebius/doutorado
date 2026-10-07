"""Renderer for `snn.normalization`'s `normalization_pipeline` frame kind:
per-feature normalization (audio: mu/sigma fit ONCE on the training set)
vs per-window normalization (EEG: mu/sigma recomputed inside every window),
and the leakage hazard that only the fitted path has (thesis chapter 07,
sec:normalizacaoEntrada).

A value marker travels RAW -> STATS -> APPLY along each branch, its number
morphing from the raw value to the normalized one on the last leg (same
construction as _paint_weight_numberline's slide). The raw boxes show the
actual data the statistics come from, so every number on screen can be
recomputed by hand.
"""

from __future__ import annotations

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, NEUTRAL_COLOR, SNN_COLOR

_AUDIO_RAW = (1.0, 5.3)
_AUDIO_FIT = (3.4, 5.3)
_AUDIO_APPLY = (5.8, 5.3)
_EEG_RAW = (1.0, 1.9)
_EEG_FIT = (3.4, 1.9)
_EEG_APPLY = (5.8, 1.9)
_LEAKY_BOX = (3.4, 3.6)
_BOX_W = 2.0
_BOX_H = 1.0


class NormalizationRendererMixin:
    def _render_normalization_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(-0.3, 7.3), ylim=(0.5, 6.7))

        for p in (_AUDIO_RAW, _AUDIO_FIT, _AUDIO_APPLY, _EEG_RAW, _EEG_FIT, _EEG_APPLY):
            self._skeleton_box(*p, w=_BOX_W, h=_BOX_H)
        self._skeleton_arrow(_AUDIO_RAW, _AUDIO_FIT)
        self._skeleton_arrow(_AUDIO_FIT, _AUDIO_APPLY)
        self._skeleton_arrow(_EEG_RAW, _EEG_FIT)
        self._skeleton_arrow(_EEG_FIT, _EEG_APPLY)

        self._ax.text(3.4, 6.55, "ÁUDIO — por característica", ha="center", fontsize=9.5, color=BITNET_COLOR, weight="bold")
        self._box(*_AUDIO_RAW, values["audio_raw_text"], NEUTRAL_COLOR, w=_BOX_W, h=_BOX_H, fontsize=8.5)
        self._box(*_AUDIO_FIT, "μ, σ: UMA vez,\nsó com o treino", BITNET_COLOR, w=_BOX_W, h=_BOX_H, fontsize=8.5)
        self._box(*_AUDIO_APPLY, "mesma régua para\nqualquer amostra", BITNET_COLOR, w=_BOX_W, h=_BOX_H, fontsize=8.5)
        self._equation_near(
            *_AUDIO_FIT, f"μ = {values['audio_mean']:g}, σ² = {values['audio_var']:g}", 1.0,
            box_h=_BOX_H, fontsize=8.5,
        )

        self._ax.text(3.4, 3.05, "EEG — por janela", ha="center", fontsize=9.5, color=SNN_COLOR, weight="bold")
        self._box(*_EEG_RAW, values["eeg_raw_text"], NEUTRAL_COLOR, w=_BOX_W, h=_BOX_H, fontsize=8.5)
        self._box(*_EEG_FIT, "μ, σ: recalculados\nnesta janela", SNN_COLOR, w=_BOX_W, h=_BOX_H, fontsize=8.5)
        self._box(*_EEG_APPLY, "régua da\nprópria janela", SNN_COLOR, w=_BOX_W, h=_BOX_H, fontsize=8.5)
        self._equation_near(
            *_EEG_FIT, f"μ = {values['eeg_mean']:g}, σ² = {values['eeg_var']:g}", 1.0,
            box_h=_BOX_H, fontsize=8.5,
        )

        self._draw_traveling_value(
            _AUDIO_RAW, _AUDIO_FIT, _AUDIO_APPLY, float(values["audio_travel"]),
            float(values["audio_x"]), float(values["audio_z"]), BITNET_COLOR,
        )
        self._draw_traveling_value(
            _EEG_RAW, _EEG_FIT, _EEG_APPLY, float(values["eeg_travel"]),
            float(values["eeg_x"]), float(values["eeg_z"]), SNN_COLOR,
        )

        drift_contrast = float(values["drift_contrast"])
        if drift_contrast > 0.02:
            self._ax.text(
                _EEG_APPLY[0], _EEG_APPLY[1] - 0.95, values["drift_text"], ha="center", va="top",
                fontsize=8.5, color=SNN_COLOR, alpha=drift_contrast, weight="bold",
            )

        leaky_reveal = float(values["leaky_reveal"])
        self._skeleton_box(*_LEAKY_BOX, w=2.8, h=0.9)
        self._skeleton_arrow(_AUDIO_RAW, _LEAKY_BOX)
        self._box(*_LEAKY_BOX, values["leaky_text"], SNN_COLOR, alpha=leaky_reveal, w=2.8, h=0.9, fontsize=8.5)

    def _draw_traveling_value(
        self, raw_pos: tuple[float, float], fit_pos: tuple[float, float], apply_pos: tuple[float, float],
        travel: float, raw_val: float, z_val: float, color: str,
    ) -> None:
        travel = max(0.0, min(1.0, travel))
        if travel <= 0.5:
            frac = travel / 0.5
            x = raw_pos[0] + (fit_pos[0] - raw_pos[0]) * frac
            y = raw_pos[1] + (fit_pos[1] - raw_pos[1]) * frac
            display = raw_val
        else:
            frac = (travel - 0.5) / 0.5
            x = fit_pos[0] + (apply_pos[0] - fit_pos[0]) * frac
            y = fit_pos[1] + (apply_pos[1] - fit_pos[1]) * frac
            display = raw_val + (z_val - raw_val) * frac
        marker_color = ACCENT_COLOR if travel >= 0.999 else color
        self._ax.plot([x], [y + 0.62], marker="o", markersize=11, color=marker_color, zorder=5)
        label = f"x' = {display:.4f}" if travel >= 0.999 else f"x = {display:.3g}"
        self._ax.text(
            x + 0.18, y + 0.62, label, ha="left", va="center", fontsize=9, color=marker_color, zorder=6, weight="bold",
        )
