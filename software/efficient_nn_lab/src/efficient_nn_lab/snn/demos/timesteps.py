"""Demonstração — time_steps × delta_t (software/nn's .wiki/Concepts/
Time-Steps.md, a página canônica de explicação didática deste projeto).

Two scenes. The movie: a playhead sweeps the 16 frames of one sample and
latency spikes light up as it reaches them; then delta_t is set against
time_steps on the same picture (a bracket under ONE frame vs the brace over
ALL of them). The tensor: the same (t, b) cells slide from time-major to
batch-major order next to a fixed "how LifBPTT reads this row" column, so
the silent failure is seen, not just stated.

Spike frames come from the encoder formula software/nn really uses,
t = round((1 - x)(T - 1)) (`latency_spike_time`), with example values chosen
away from the .5 rounding boundaries (0.9 would give exactly 1.5, where float
precision decides the frame); the wiki diagram uses the same 0.8/0.4/0.2.
"""

from __future__ import annotations

import math

from efficient_nn_lab.app.theme import ACCENT_COLOR, BITNET_COLOR, SNN_COLOR
from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence, slider, transition
from efficient_nn_lab.snn.encoding import latency_spike_time

_T = 16
_SMALL_T = 4
_RC = 5.0  # R*C of snn.lif's default neuron (R = 5, C = 1)
_EXAMPLES = ((0.8, SNN_COLOR), (0.4, ACCENT_COLOR), (0.2, BITNET_COLOR))
_EXAMPLE_SPIKES = [(latency_spike_time(x, _T), x, color) for x, color in _EXAMPLES]
_MOVIE_EQUATION = "t_spike = \\text{round}((1 - x)(T - 1))"
_BETA_EQUATION = "beta = e^{-\\dfrac{Δt}{R C}}"


class TimeStepsDemo(DemoModule):
    title = "SNN -> time_steps x delta_t"
    slug = "snn.timesteps"
    description = (
        "time_steps (quantos quadros) e delta_t (quanto dura cada quadro) parecem o mesmo "
        "nome, mas significam coisas completamente diferentes -- confundi-los produz uma rede "
        "que treina, mas não aprende nada temporal."
    )

    def __init__(self) -> None:
        self.n_samples = 2
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {"n_samples": slider("B (amostras empilhadas)", 2.0, 3.0, 1.0, float(self.n_samples))}

    def _build_frames(self) -> list[Frame]:
        return self._movie_frames() + self._tensor_frames()

    def _movie_frames(self) -> list[Frame]:
        (f_hi, x_hi, _), (f_mid, x_mid, _), (f_lo, x_lo, _) = _EXAMPLE_SPIKES
        near_lo = 0.22
        stops = {
            0: (
                "Por que um neurônio de pulso precisa de tempo",
                "Um neurônio comum é uma função: um número entra, outro sai. Um neurônio de pulso tem "
                "memória: o potencial $V_t$ soma a entrada de agora ao que sobrou de $V_{t-1}$, e ele só "
                "dispara quando acumula o bastante. Uma foto única não basta — cada amostra vira um "
                f"filme curto. Aqui o filme tem {_T} quadros (0 a {_T - 1}); acompanhe o cursor.",
            ),
            f_hi: (
                f"x = {x_hi:.1f} dispara no quadro {f_hi}",
                "Codificação por latência: o valor de uma feature vira o MOMENTO do seu disparo — o "
                f"quadro (1 − x) × (T − 1), arredondado. Para x = {x_hi:.1f}: (1 − {x_hi:.1f}) × {_T - 1} "
                f"= {f_hi}, então dispara no quadro {f_hi}. Quanto maior o valor, mais cedo o disparo.",
            ),
            f_mid: (
                f"x = {x_mid:.1f} dispara no quadro {f_mid}",
                f"x = {x_mid:.1f}: (1 − {x_mid:.1f}) × {_T - 1} = {f_mid}, quadro {f_mid}. Todo valor gera "
                "exatamente UM disparo: a informação não está em quantos pulsos há, e sim em QUANDO ele "
                f"acontece. Por isso a ordem dos quadros importa — o disparo no quadro {f_mid} só significa "
                f"{x_mid:.1f} porque vem depois de {f_mid} quadros em silêncio.",
            ),
            f_lo: (
                f"x = {x_lo:.1f} dispara no quadro {f_lo}",
                f"x = {x_lo:.1f}: (1 − {x_lo:.1f}) × {_T - 1} = {f_lo}, quadro {f_lo}; x = 0 dispararia no "
                f"último, o {_T - 1}. Com {_T} quadros só existem {_T} valores possíveis: x = {x_lo:.1f} e "
                f"x = {near_lo} caem ambos no quadro {latency_spike_time(near_lo, _T)}. Discretizar o tempo "
                "custa precisão — é o chão de ruído da latência (demo Ruído estrutural).",
            ),
            _T - 1: (
                f"time_steps = {_T}: QUANTOS quadros",
                f"time_steps = {_T} responde QUANTOS quadros cada amostra ocupa: {_T} linhas no tensor, "
                f"não uma, e {_T} passos para o BPTT percorrer de trás para frente. No software/nn ele "
                "não tem valor padrão: deixar sem definir é erro, porque assumir 1 criaria em silêncio "
                "uma 'rede de pulso' de um passo só — sem memória, sem aprendizado temporal, mas que "
                "treina e mostra uma perda.",
            ),
        }

        frames: list[Frame] = []
        for t in range(_T):
            label, explanation = stops[min(s for s in stops if s >= t)]
            frames.append(
                Frame(
                    label=label,
                    values=self._movie_values(float(t), dt_reveal=0.0),
                    explanation=explanation,
                    equation=_MOVIE_EQUATION,
                    is_checkpoint=t in stops,
                )
            )

        beta_short = math.exp(-1.0 / _RC)
        beta_long = math.exp(-float(_T) / _RC)
        delta_t = Frame(
            label="delta_t: QUANTO DURA cada quadro",
            values=self._movie_values(float(_T - 1), dt_reveal=1.0),
            explanation=(
                "delta_t (colchete) responde outra pergunta: QUANTO DURA um quadro. Ele só entra em "
                "beta, a fração do potencial que sobrevive de um quadro para o seguinte. Com R·C = "
                f"{_RC:g}: delta_t = 1 dá $beta = {beta_short:.2f}$ (lembra bem); delta_t = {_T} dá "
                f"$beta = {beta_long:.2f}$ (esquece quase tudo). Pôr {_T} em delta_t achando que são "
                f"{_T} quadros não dá erro nenhum: só apaga a memória."
            ),
            equation=_BETA_EQUATION,
        )
        frames += transition(frames[-1], delta_t, steps=8)
        frames.append(delta_t)
        return frames

    @staticmethod
    def _movie_values(playhead: float, dt_reveal: float) -> dict[str, object]:
        return {
            "kind": "timesteps_tensor",
            "stage": "movie",
            "time_steps": _T,
            "playhead": playhead,
            "dt_reveal": dt_reveal,
            "example_spikes": _EXAMPLE_SPIKES,
        }

    def _tensor_frames(self) -> list[Frame]:
        n_samples = int(self.n_samples)
        total_rows = _SMALL_T * n_samples
        cell_h = 0.72 / total_rows
        slot_y = [0.86 - (r + 0.5) * cell_h for r in range(total_rows)]
        cells = [(t, b) for t in range(_SMALL_T) for b in range(n_samples)]

        def checkpoint(row_of, label: str, title: str, explanation: str, equation: str) -> Frame:
            return Frame(
                label=label,
                values={
                    "kind": "timesteps_tensor",
                    "stage": "tensor",
                    "total_rows": total_rows,
                    "n_samples": n_samples,
                    "slot_y": slot_y,
                    "row_y": [slot_y[row_of(t, b)] for t, b in cells],
                    "row_t": [t for t, _ in cells],
                    "row_sample": [b for _, b in cells],
                    "order_title": title,
                },
                explanation=explanation,
                equation=equation,
            )

        correct = checkpoint(
            lambda t, b: t * n_samples + b,
            "O tensor é plano: só time_steps diz onde cada amostra acaba",
            "Ordem correta (time-major): linha = t·B + b",
            f"Para a camada, o lote vira uma matriz 2-D sem fronteiras: T = {_SMALL_T} quadros × "
            f"B = {n_samples} amostras = {total_rows} linhas. São {total_rows} amostras de 1 quadro? "
            f"{n_samples} de {_SMALL_T}? {_SMALL_T} de {n_samples}? A matriz sozinha não diz; "
            f"time_steps diz: B = {total_rows}/{_SMALL_T} = {n_samples}. Na ordem correta, todas as "
            "amostras do quadro 0 vêm primeiro, depois todas do quadro 1 (linha = t·B + b). À direita, "
            "como o LifBPTT lê cada linha: tudo confere.",
            "row = t · B + b \\text{  (correto: time-major)}",
        )
        wrong = checkpoint(
            lambda t, b: b * _SMALL_T + t,
            "Ordem errada (batch-major): roda sem erro, aprende errado",
            "Ordem errada (batch-major): linha = b·T + t",
            "Empilhar por amostra (linha = b·T + t) não muda nenhum valor, só a ordem — veja as "
            "células deslizarem. Mas o LifBPTT continua lendo a linha 1 como (t=0, b=1) e recebe "
            "(t=1, b=0): o 2º quadro da amostra 0 entra como 1º quadro da amostra 1. Nenhum erro, "
            "nenhum aviso, e a perda até cai — a rede só aprende uma ordem temporal embaralhada. "
            "Falha silenciosa.",
            "row = b · T + t \\text{  (errado: batch-major)}",
        )
        return build_sequence([correct, wrong], steps=12)
