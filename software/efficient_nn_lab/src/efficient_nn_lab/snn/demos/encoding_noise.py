"""Demonstração — ruído estrutural das 3 codificações, antes de qualquer
aprendizado (software/nn's .wiki/Concepts/Spike-Encoding.md). Explica por
que "direta > latência > Poisson" em reconstrução pode ser apenas esse
piso de ruído, não uma diferença real de informação carregada.

The cursor sweeps T upward; at the slider's T the spotlight cross-fades
from the Poisson curve to latency to direct (tweened 0..1 weights, so it
glides rather than snaps), each with its own mechanism and numbers; then
the sweep continues to T=64 for the cost/benefit and the fair-comparison
conclusion.

Both curves are the SAME statistic -- the RMS error with x spread uniformly
over [0, 1] -- so they can honestly share one axis: sqrt(1/(6T)) for
Poisson, spacing/sqrt(12) = 1/((T-1) sqrt(12)) for latency (the encoder
software/nn really runs, `latency_spike_time`). An earlier version plotted
the Poisson standard deviation at its worst-case p against the latency
WORST-CASE error -- a typical value against a maximum on one axis. The
worst cases still appear in the text, labelled as such.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, slider, transition
from efficient_nn_lab.snn.encoding import (
    latency_quantization_error,
    latency_rms_error,
    poisson_noise_sigma,
    poisson_rms_error,
)

_T_MIN, _T_MAX = 4, 64
_FOCUS_STEPS = 6
_EQUATION = (
    "RMS_poisson = \\sqrt{\\dfrac{1}{6T}};  RMS_latencia = \\dfrac{1}{(T - 1)\\sqrt{12}};  erro_direta = 0"
)


class EncodingNoiseDemo(DemoModule):
    title = "SNN -> Ruído estrutural da codificação"
    slug = "snn.encoding_noise"
    description = (
        "Antes de qualquer aprendizado, cada codificação de uma SNN (Spiking Neural Network, "
        "rede neural de pulso) já injeta um piso de ruído estrutural diferente. 'Direta > "
        "latência > Poisson' em reconstrução pode ser só esse piso de ruído, não uma diferença "
        "real de informação."
    )

    def __init__(self) -> None:
        self.time_steps = 16
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {"time_steps": slider("T (time_steps)", float(_T_MIN), float(_T_MAX), 1.0, float(self.time_steps))}

    def _build_frames(self) -> list[Frame]:
        t_range = np.arange(_T_MIN, _T_MAX + 1)
        curves = {
            "kind": "encoding_noise_floor",
            "t_range": t_range,
            "poisson_curve": np.array([poisson_rms_error(int(t)) for t in t_range]),
            "latency_curve": np.array([latency_rms_error(int(t)) for t in t_range]),
        }
        t_sel = int(round(self.time_steps))
        p_sel, l_sel = poisson_rms_error(t_sel), latency_rms_error(t_sel)
        p_worst, l_worst = poisson_noise_sigma(t_sel), latency_quantization_error(t_sel)

        def values(t: int, w_p: float = 1.0, w_l: float = 1.0, w_d: float = 1.0) -> dict[str, object]:
            return {
                **curves,
                "t_current": t,
                "poisson_at_t": poisson_rms_error(t),
                "latency_at_t": latency_rms_error(t),
                "w_poisson": w_p,
                "w_latency": w_l,
                "w_direct": w_d,
            }

        def frame(t: int, label: str, explanation: str, checkpoint: bool, **weights: float) -> Frame:
            return Frame(label, values(t, **weights), explanation, _EQUATION, is_checkpoint=checkpoint)

        intro = (
            "Antes de aprender qualquer coisa",
            "Três SNNs reconstroem o mesmo sinal, cada uma com uma codificação. Nem uma rede perfeita "
            "reconstrói melhor do que a própria codificação permite: cada uma já injeta um erro "
            f"mínimo — um piso — antes de qualquer treino. Com T = {_T_MIN} quadros o piso é alto; "
            "acompanhe o cursor aumentando T. As duas curvas usam a mesma régua: o erro RMS típico, "
            "com o valor x espalhado por igual em [0, 1].",
        )
        poisson = (
            "Poisson: o piso é estatístico",
            "Poisson: em cada quadro a unidade dispara com probabilidade p = x, e o valor é lido "
            "como disparos/T. A contagem é sorteada: duas passadas do mesmo x dão contagens "
            f"diferentes. O desvio dessa leitura é √(x(1 − x)/T): no pior caso, x = 0.5, vale "
            f"{p_worst:.3f} em T = {t_sel}. Na média sobre x, o erro RMS é √(1/(6T)): "
            f"$RMS = {p_sel:.3f}$ — um erro típico de {100 * p_sel:.1f}% da escala inteira.",
        )
        latency = (
            "Latência: o piso é de arredondamento",
            f"Latência: não há sorteio, mas só existem T = {t_sel} momentos de disparo, ou seja {t_sel} "
            f"níveis espaçados de 1/(T − 1) = {1 / (t_sel - 1):.3f}. Um valor entre dois níveis vira o "
            f"mais próximo: no pior caso erra meio espaço, 0.5/(T − 1) = {l_worst:.4f}; na média, o erro "
            f"RMS é espaço/√12 = {l_sel:.4f} — a mesma régua da curva Poisson. Sempre o mesmo resultado, "
            "mas em degraus.",
        )
        direct = (
            "Direta: piso zero",
            "Direta: o valor analógico entra como corrente, sem virar pulsos na entrada — nada a "
            f"sortear, nada a arredondar: piso zero (a linha no eixo). Em T = {t_sel}, erro RMS: direta 0 "
            f"< latência {l_sel:.3f} < Poisson {p_sel:.3f}. Se um experimento mostra 'direta > "
            "latência > Poisson', essa ordem pode ser só a dos pisos, não informação a mais.",
        )
        more_t = (
            "Mais quadros baixam o piso — a um custo",
            "Mais quadros baixam os dois pisos, em ritmos diferentes: de T = 16 para T = 64 (4×), "
            f"Poisson só cai pela metade ({poisson_rms_error(16):.3f} → {poisson_rms_error(64):.3f}, "
            "∝ 1/√T); latência cai uns 4× ("
            f"{latency_rms_error(16):.4f} → {latency_rms_error(64):.4f}, ∝ 1/T). "
            "Nenhum chega a zero, e cada quadro a mais custa tempo de simulação e energia.",
        )
        fair = (
            "Como comparar codificações com justiça",
            "A armadilha: comparar o erro de reconstrução das três e concluir que uma 'carrega mais "
            "informação'. Parte da diferença é só o piso de cada codificação — uma falha silenciosa "
            "de interpretação, sem erro nenhum no código. O justo é medir quanto cada rede fica "
            "ACIMA do seu próprio piso, no mesmo T e com a mesma estatística de erro.",
        )

        frames = [frame(_T_MIN, *intro, checkpoint=True)]
        for t in range(_T_MIN + 1, t_sel):
            frames.append(frame(t, *poisson, checkpoint=False))
        focus = [
            frame(t_sel, *poisson, checkpoint=True, w_l=0.0, w_d=0.0),
            frame(t_sel, *latency, checkpoint=True, w_p=0.0, w_d=0.0),
            frame(t_sel, *direct, checkpoint=True, w_p=0.0, w_l=0.0),
        ]
        frames += transition(frames[-1], focus[0], _FOCUS_STEPS)
        frames.append(focus[0])
        for a, b in zip(focus, focus[1:]):
            frames += transition(a, b, _FOCUS_STEPS)
            frames.append(b)
        back_to_all = frame(t_sel, *more_t, checkpoint=False)
        frames += transition(focus[-1], back_to_all, _FOCUS_STEPS)
        frames.append(back_to_all)
        for t in range(t_sel + 1, _T_MAX):
            frames.append(frame(t, *more_t, checkpoint=False))
        frames.append(frame(_T_MAX, *more_t, checkpoint=True))
        frames.append(frame(_T_MAX, *fair, checkpoint=True))
        return frames
