"""Demonstração — normalização dependente de limiar (tdBN): bruto ->
BatchNorm comum -> tdBN, com o exemplo numérico do software/nn
(X = [0, 2, 4, 6], mu = 3, sigma^2 = 5 -- .wiki/Concepts/
Threshold-Dependent-Batch-Normalization.md).

The default threshold is V_th = 2 on purpose: at V_th = 1 plain BatchNorm
and tdBN produce the same numbers and the demo would have nothing to show.
The last step sweeps the threshold itself: the tdBN dots scale with it and
the count above threshold never changes, while a ghost row of plain
BatchNorm values (which never move) goes silent.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame, slider, transition
from efficient_nn_lab.snn.tdbn import tdbn_transform

_X = np.array([0.0, 2.0, 4.0, 6.0])  # one channel, T = 2 frames x B = 2 samples: t0b0, t0b1, t1b0, t1b1
_V_TH_MIN, _V_TH_MAX = 1.0, 4.0
_STEPS = 12
_EQUATION = "Y = alpha · V_th · \\dfrac{X - mu}{\\sqrt{sigma^2 + \\epsilon}}"


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in values) + "]"


def _pass(n: int) -> str:
    return f"{n} de 4 {'passa' if n == 1 else 'passam'}"


class TdBNDemo(DemoModule):
    title = "SNN -> Normalização dependente de limiar (tdBN)"
    slug = "snn.tdbn"
    description = (
        "tdBN normaliza a corrente de entrada de um neurônio LIF para um espalhamento "
        "proporcional ao próprio limiar de disparo V_th, em vez de variância unitária."
    )

    def __init__(self) -> None:
        self.v_th = 2.0
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {"v_th": slider("V_th (limiar)", _V_TH_MIN, _V_TH_MAX, 0.5, self.v_th)}

    def _build_frames(self) -> list[Frame]:
        v = float(self.v_th)
        x_hat = tdbn_transform(_X, v_th=1.0)  # plain BatchNorm: zero mean, unit variance
        y = tdbn_transform(_X, v_th=v)
        target = _V_TH_MAX if v <= 2.5 else _V_TH_MIN
        y_target = tdbn_transform(_X, v_th=target)
        span = max(float(np.abs(_X).max()), float(np.abs(tdbn_transform(_X, v_th=_V_TH_MAX)).max())) * 1.18
        mu, std = float(_X.mean()), float(_X.std())

        def fires(values: np.ndarray, threshold: float) -> int:
            return int(np.sum(values > threshold))

        def frame(label: str, explanation: str, **values: object) -> Frame:
            base = {
                "kind": "tdbn_distribution", "x": _X, "v_th": v, "span": span,
                "ghost_y": x_hat, "ghost_reveal": 0.0,
            }
            return Frame(label, {**base, **values}, explanation, _EQUATION)

        raw = frame(
            "Corrente bruta: a escala não conversa com o limiar",
            "Um LIF só dispara se a corrente empurra V acima de V_th. Aqui, 4 correntes de um mesmo "
            "canal: 2 amostras × 2 quadros, juntadas — tdBN calcula média e desvio sobre o lote E o "
            f"tempo. X = [0, 2, 4, 6]; com V_th = {v:g}, {_pass(fires(_X, v))}. Mas isso é sorte "
            "da escala: numa camada mais funda X pode encolher até ninguém disparar, ou crescer até "
            "todos dispararem.",
            y=_X, main_label="X bruto", stage_title="1. Corrente bruta",
        )
        n_bn = fires(x_hat, v)
        bn_verdict = (
            f"Com V_th = {v:g}, NENHUM valor passa: a camada fica muda. Sem disparo o gradiente "
            "substituto é ≈ 0 e ela não aprende — o 'No-Spike Problem'."
            if n_bn == 0
            else f"Com V_th = {v:g}, {_pass(n_bn)} — só porque este limiar está perto do 1 que a "
            "variância 1 supõe. Suba V_th no controle e a camada emudece."
        )
        bn = frame(
            "BatchNorm comum: variância 1, cego ao limiar",
            f"BatchNorm comum padroniza: Y = (X − {mu:g})/{std:.2f} = {_fmt(x_hat)} — média 0, "
            f"variância 1, qualquer que seja a escala de entrada. {bn_verdict}",
            y=x_hat, main_label="BatchNorm comum", stage_title="2. BatchNorm comum: Y = (X − μ)/σ",
        )
        td = frame(
            "tdBN: espalhamento α·V_th",
            f"tdBN multiplica a versão padronizada por α·V_th (aqui α = 1): Y = {v:g}·(X − {mu:g})/"
            f"{std:.2f} = {_fmt(y)}. O desvio de Y passa a ser exatamente V_th = {v:g}, e "
            f"{_pass(fires(y, v))} do limiar. A régua da normalização acompanha o limiar do neurônio.",
            y=y, main_label="tdBN", stage_title="3. tdBN: Y = α·V_th·(X − μ)/σ",
        )
        sweep = frame(
            "Mude o limiar: tdBN acompanha, BatchNorm não",
            f"O limiar vai de {v:g} para {target:g}: os pontos do tdBN escalam junto e a contagem acima "
            f"de V_th não muda — {_pass(fires(y_target, target))}. A fileira cinza é o BatchNorm comum, "
            f"que ignora V_th: {_pass(fires(x_hat, target))}. Em geral, P(Y > V_th) = P(X̂ > 1/α) não "
            "depende de V_th, da escala nem da profundidade — com α = 1 e entradas ~gaussianas, "
            "≈ 15.9% por passo (Zheng et al.).",
            y=y_target, v_th=target, ghost_reveal=1.0, main_label="tdBN",
            stage_title="4. O mesmo tdBN com outro limiar",
        )

        frames = [raw]
        for a, b in ((raw, bn), (bn, td), (td, sweep)):
            frames += transition(a, b, _STEPS)
            frames.append(b)
        return frames
