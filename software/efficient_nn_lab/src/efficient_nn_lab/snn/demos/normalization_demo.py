"""Demonstração — normalização de entrada por característica (áudio,
ajustada UMA vez no treino) × por janela (EEG, recalculada a cada janela),
e o vazamento que só o caminho ajustado tem (tese, capítulo 07,
sec:normalizacaoEntrada, incluindo o exemplo numérico de lá).

Every statistic is computed from the thesis's own example data (the audio
training column [2, 4, 4, 6], the EEG window [10, 12, 14, 16] uV and the
same window shifted by +50 uV), not typed in, so the screen and the thesis
cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, transition
from efficient_nn_lab.snn.normalization import fit_zscore, leaky_fit_zscore, zscore

_AUDIO_TRAIN = np.array([2.0, 4.0, 4.0, 6.0])
_AUDIO_TEST_X = 5.0
_EEG_WINDOW = np.array([10.0, 12.0, 14.0, 16.0])
_EEG_DRIFT_UV = 50.0
_STEPS = 10
_EQUATION = "x' = \\dfrac{x - mu}{sigma}"


def _fmt_list(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


class NormalizationDemo(DemoModule):
    title = "SNN -> Normalização: por característica x por janela"
    slug = "snn.normalization"
    description = (
        "Áudio normaliza cada característica com média/desvio ajustados UMA vez no treino (CMVN); "
        "EEG recalcula média/desvio dentro de cada janela. Ajustar com dados de teste é um "
        "vazamento silencioso -- e só o caminho ajustado corre esse risco."
    )

    def _build_frames(self) -> list[Frame]:
        a_mean, a_std = fit_zscore(_AUDIO_TRAIN)
        a_z = float(zscore(_AUDIO_TEST_X, a_mean, a_std))
        e_mean, e_std = fit_zscore(_EEG_WINDOW)
        e_x = float(_EEG_WINDOW[0])
        e_z = float(zscore(e_x, e_mean, e_std))
        drifted = _EEG_WINDOW + _EEG_DRIFT_UV
        d_mean, d_std = fit_zscore(drifted)
        d_x = float(drifted[0])
        d_z = float(zscore(d_x, d_mean, d_std))
        frozen_z = float(zscore(d_x, e_mean, e_std))  # the audio strategy applied to EEG
        l_mean, l_std = leaky_fit_zscore(_AUDIO_TRAIN, np.array([_AUDIO_TEST_X]))
        l_z = float(zscore(_AUDIO_TEST_X, l_mean, l_std))

        base = {
            "kind": "normalization_pipeline",
            "audio_raw_text": f"coluna j, treino:\n{_fmt_list(_AUDIO_TRAIN)}",
            "audio_x": _AUDIO_TEST_X,
            "audio_mean": round(a_mean, 4),
            "audio_var": round(a_std**2, 4),
            "audio_z": a_z,
            "eeg_raw_text": f"janela i:\n{_fmt_list(_EEG_WINDOW)} µV",
            "eeg_x": e_x,
            "eeg_mean": round(e_mean, 4),
            "eeg_var": round(e_std**2, 4),
            "eeg_z": e_z,
            "drift_text": f"com μ, σ fixos da janela anterior: x' = {frozen_z:.1f}  (errado)",
            "leaky_text": f"ERRADO: ajustar com treino + teste\nμ = {l_mean:.1f}, σ = {l_std:.2f} → x' = {l_z:.2f}",
            "audio_travel": 0.0,
            "eeg_travel": 0.0,
            "drift_contrast": 0.0,
            "leaky_reveal": 0.0,
        }

        def frame(label: str, explanation: str, checkpoint: bool = True, **overrides: object) -> Frame:
            return Frame(label, {**base, **overrides}, explanation, _EQUATION, is_checkpoint=checkpoint)

        problem = frame(
            "Escalas diferentes atrapalham o gradiente",
            "Não dá para somar metros com quilogramas; uma rede tem o mesmo problema com entradas em "
            "escalas diferentes: a característica que varia até 10 000 domina o gradiente e a que varia "
            "até 1 é ignorada. A cura é padronizar: x' = (x − μ)/σ. Mas μ e σ calculados sobre O QUÊ? "
            "Áudio e EEG respondem diferente.",
        )
        audio = frame(
            "Áudio: por característica, ajustado uma vez",
            "No áudio, a coluna j é sempre a mesma banda de frequência: sua escala é da característica, "
            f"não da amostra. Então μ e σ saem UMA vez, só do treino {_fmt_list(_AUDIO_TRAIN)}: μ = "
            f"{a_mean:g}, σ² = {a_std**2:g}. Uma amostra de teste x = {_AUDIO_TEST_X:g}, nunca vista no "
            f"ajuste, usa a mesma régua: $x' = {a_z:.4f}$. É o CMVN do reconhecimento de fala.",
            audio_travel=1.0,
        )
        eeg = frame(
            "EEG: por janela, recalculado sempre",
            "No EEG a amplitude não é estável: impedância dos eletrodos, ganho do amplificador e a "
            "linha de base mudam de sessão para sessão. Então cada janela calcula a PRÓPRIA régua, na "
            f"hora: {_fmt_list(_EEG_WINDOW)} µV dá μ = {e_mean:g}, σ² = {e_std**2:g}, e o ponto "
            f"x = {e_x:g} vira $x' = {e_z:.4f}$. O custo: perde-se a amplitude absoluta — pouco confiável "
            "no EEG bruto.",
            audio_travel=1.0,
            eeg_travel=1.0,
        )
        drift_label = f"A janela seguinte chega com +{_EEG_DRIFT_UV:g} µV de deriva"
        drift_explanation = (
            f"Mesma atividade, linha de base +{_EEG_DRIFT_UV:g} µV (comum entre sessões): "
            f"{_fmt_list(drifted)}. A janela recalcula: μ = {d_mean:g}, σ² = {d_std**2:g}, e "
            f"x = {d_x:g} vira de novo $x' = {d_z:.4f}$ — a deriva sumiu. Com a régua fixa do áudio "
            f"(μ = {e_mean:g}, σ = {e_std:.2f}) daria {frozen_z:.1f}: a rede veria um valor absurdo "
            "que é só artefato da sessão."
        )
        new_window = {
            "audio_travel": 1.0,
            "eeg_raw_text": f"janela i+1 (+{_EEG_DRIFT_UV:g} µV):\n{_fmt_list(drifted)} µV",
            "eeg_x": d_x,
            "eeg_mean": round(d_mean, 4),
            "eeg_var": round(d_std**2, 4),
            "eeg_z": d_z,
        }
        drift_start = frame(drift_label, drift_explanation, checkpoint=False, **new_window)
        drift = frame(drift_label, drift_explanation, **new_window, eeg_travel=1.0, drift_contrast=1.0)
        leak = frame(
            "Vazamento: ajustar a régua com o teste",
            "Estudar para a prova já sabendo as respostas: se μ e σ do áudio forem ajustados com treino "
            f"+ teste, o próprio x = {_AUDIO_TEST_X:g} de teste puxa a régua — μ = {l_mean:.1f}, "
            f"σ = {l_std:.2f}, x' = {l_z:.2f} em vez de {a_z:.2f}. Nenhum erro aparece; só as métricas "
            "de validação ficam otimistas demais. O EEG por janela não ajusta nada: não tem como vazar.",
            **{**drift.values, "leaky_reveal": 1.0},
        )

        frames = [problem]
        for a, b in ((problem, audio), (audio, eeg)):
            frames += transition(a, b, _STEPS)
            frames.append(b)
        frames.append(drift_start)  # jump: a new window arrives, marker back at RAW
        frames += transition(drift_start, drift, _STEPS)
        frames.append(drift)
        frames += transition(drift, leak, _STEPS)
        frames.append(leak)
        return frames
