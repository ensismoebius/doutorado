"""Demonstração — Forward e backward tradicionais, até convergir.

Duas partes, um único fio condutor: primeiro o mecanismo (o que o forward
calcula, o que a regra da cadeia calcula no backward, como o peso é
atualizado uma única vez); depois o resultado de repetir esse mesmo passo
várias vezes seguidas — um neurônio de verdade convergindo para um alvo,
número por número, até a saída ficar arbitrariamente perto do que se
queria.

Nenhuma quantização aqui: é o caso "liso" (sem função em degrau no meio)
que as demonstrações de STE (bitnet/demos/backward.py) e de surrogate
gradient (snn/demos/surrogate_gradient.py) modificam. Ver esta primeiro
deixa claro exatamente o que aquelas duas estão contornando.

Reaproveita deliberadamente `bitnet.linear.squared_error_loss` /
`loss_gradient_wrt_y` e `bitnet.ste.sgd_update` — são matemática genérica
de regressão/gradiente descendente, não específica de BitNet; duplicá-las
aqui seria a mesma fórmula com um nome diferente.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.bitnet.linear import loss_gradient_wrt_y, squared_error_loss
from efficient_nn_lab.bitnet.ste import sgd_update
from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.core.math_utils import ease_in_out, lerp

#: Interior sliding frames generated per gradient-descent iteration in the
#: convergence chart — enough for the point's motion toward its new value
#: to read as a clear slide, not a jump.
_SUBSTEPS_PER_ITERATION = 10


class TraditionalBackpropDemo(DemoModule):
    title = "Backprop -> Forward e backward classicos"
    description = (
        "Como o forward e o backward funcionam sem nenhum truque no meio — e o que "
        "acontece quando se repete esse passo várias vezes: o peso converge para o "
        "valor que faz a saída bater com o alvo."
    )

    def __init__(self) -> None:
        self.x = 2.0
        self.target = 6.0
        self.w_init = 0.1
        self.learning_rate = 0.1
        self.n_iterations = 10
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "target": {"label": "Alvo", "min": 1.0, "max": 10.0, "step": 0.5, "value": self.target},
            "learning_rate": {"label": "Taxa de aprendizado", "min": 0.02, "max": 0.2, "step": 0.02, "value": self.learning_rate},
        }

    # -- part 1: the mechanism, once, explained ----------------------------
    def _build_pipeline_checkpoints(self) -> list[Frame]:
        x, w = self.x, self.w_init
        y = w * x
        diff = y - self.target
        loss = squared_error_loss(y, self.target)
        grad_y = loss_gradient_wrt_y(y, self.target)
        grad_w = grad_y * x
        w_updated = sgd_update(w, grad_w, self.learning_rate)

        base = {
            "kind": "backprop_pipeline",
            "x": x, "w": w, "y": y, "target": self.target, "diff": diff, "loss": loss,
            "grad_y": grad_y, "grad_w": grad_w, "w_updated": w_updated, "lr": self.learning_rate,
            "y_reveal": 0.0, "target_reveal": 0.0, "diff_reveal": 0.0, "loss_reveal": 0.0,
            "grady_reveal": 0.0, "gradw_reveal": 0.0, "update_reveal": 0.0, "w_pulse": 0.0,
        }

        def frame(label: str, explanation: str, equation: str = "", **overrides) -> Frame:
            values = dict(base)
            values.update(overrides)
            return Frame(label, values, explanation, equation)

        return [
            frame(
                "O neurônio",
                "Um neurônio de um peso só, sem quantização: y = w * x. Primeiro o "
                "forward calcula y; depois o backward calcula o quanto mudar w.",
                equation="y = w * x",
            ),
            frame(
                "Forward: calcular y",
                f"O forward é só isso: multiplicar. w = {w:g}, x = {x:g}, então y = {y:g}.",
                equation=f"y = w * x = {w:g} * {x:g} = {y:g}",
                y_reveal=1.0,
            ),
            frame(
                "Comparar com o alvo",
                f"O alvo é {self.target:g}; a saída atual é {y:g} — ainda longe. Essa "
                "diferença é o que o treino tenta reduzir a cada passo.",
                y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0,
            ),
            frame(
                "A perda (loss)",
                "A perda resume a diferença em um único número, sempre positivo, que cresce "
                "quanto mais longe do alvo a saída estiver.",
                equation="L = 1/2 (y - target)^2",
                y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0,
            ),
            frame(
                "Backward, primeiro passo: dL/dy",
                f"O backward começa no fim: o quanto a perda muda se y mudasse um "
                f"pouco. dL/dy = y - target = {grad_y:g}.",
                equation="dL/dy = y - target",
                y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0, grady_reveal=1.0,
            ),
            frame(
                "Backward, regra da cadeia: dL/dw",
                f"Como y = w * x, uma mudança em w se propaga para y multiplicada por x. A "
                f"regra da cadeia é exatamente essa multiplicação: dL/dw = dL/dy * x = {grad_w:g}.",
                equation="dL/dw = dL/dy * dy/dw = dL/dy * x",
                y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0,
                grady_reveal=1.0, gradw_reveal=1.0,
            ),
            frame(
                "Atualizar o peso",
                f"O peso anda um pequeno passo no sentido contrário ao gradiente: "
                f"w vira {w_updated:g}, um pouco mais perto do valor que zeraria a perda. "
                "Repetir este passo várias vezes é exatamente a próxima parte.",
                equation="w <- w - taxa * dL/dw",
                y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0,
                grady_reveal=1.0, gradw_reveal=1.0, update_reveal=1.0, w_pulse=1.0,
            ),
        ]

    # -- part 2: the same step, repeated, to convergence --------------------
    def _run_gradient_descent(self) -> tuple[list[float], list[float], list[float]]:
        w = self.w_init
        w_hist, y_hist, loss_hist = [], [], []
        for _ in range(self.n_iterations + 1):
            y = w * self.x
            loss = squared_error_loss(y, self.target)
            w_hist.append(w)
            y_hist.append(y)
            loss_hist.append(loss)
            grad_y = loss_gradient_wrt_y(y, self.target)
            grad_w = grad_y * self.x
            w = sgd_update(w, grad_w, self.learning_rate)
        return w_hist, y_hist, loss_hist

    def _build_convergence_frames(self) -> list[Frame]:
        w_hist, y_hist, loss_hist = self._run_gradient_descent()
        n = self.n_iterations
        y_min = min(0.0, min(y_hist), self.target) - 0.5
        y_max = max(max(y_hist), self.target) + 0.5
        loss_min = max(1e-4, min(loss_hist) * 0.5)
        loss_max = max(loss_hist) * 1.5

        def chart_values(k_exact: int, w_val: float, y_val: float, loss_val: float) -> dict[str, object]:
            return {
                "kind": "backprop_convergence",
                "iterations": np.arange(k_exact + 1),
                "w": np.array(w_hist[:k_exact] + [w_val]),
                "y": np.array(y_hist[:k_exact] + [y_val]),
                "loss": np.array(loss_hist[:k_exact] + [loss_val]),
                "target": self.target,
                "n_total": n,
                "y_min": y_min, "y_max": y_max,
                "loss_min": loss_min, "loss_max": loss_max,
            }

        frames: list[Frame] = [
            Frame(
                "Antes de treinar (iteração 0)",
                chart_values(0, w_hist[0], y_hist[0], loss_hist[0]),
                f"Peso inicial w = {w_hist[0]:g}, bem longe do que faria y bater com o "
                f"alvo {self.target:g}. Cada iteração daqui pra frente repete exatamente o "
                "forward + backward + atualização da parte anterior.",
                is_checkpoint=True,
            )
        ]
        for k in range(1, n + 1):
            for s in range(1, _SUBSTEPS_PER_ITERATION + 1):
                t = ease_in_out(s / (_SUBSTEPS_PER_ITERATION + 1))
                frames.append(
                    Frame(
                        f"Iteração {k}",
                        chart_values(
                            k - 1,
                            lerp(w_hist[k - 1], w_hist[k], t),
                            lerp(y_hist[k - 1], y_hist[k], t),
                            lerp(loss_hist[k - 1], loss_hist[k], t),
                        ),
                        "",
                        is_checkpoint=False,
                    )
                )
            diff = abs(y_hist[k] - self.target)
            explanation = (
                f"Depois de {k} iteração(ões): w = {w_hist[k]:.3f}, y = {y_hist[k]:.3f}, "
                f"a {diff:.3f} de distância do alvo."
            )
            if k == n:
                explanation += " A saída já está tão perto do alvo que a diferença some visualmente — é isso que 'convergir' significa aqui."
            frames.append(
                Frame(
                    f"Iteração {k}",
                    chart_values(k, w_hist[k], y_hist[k], loss_hist[k]),
                    explanation,
                    is_checkpoint=True,
                )
            )
        return frames

    def _build_frames(self) -> list[Frame]:
        pipeline = build_sequence(self._build_pipeline_checkpoints(), steps=14)
        convergence = self._build_convergence_frames()
        # deliberate cut: block diagram -> line chart is a genuine change
        # of visualization kind, not a blend of unrelated pictures.
        bridge = Frame(
            convergence[0].label,
            convergence[0].values,
            "Agora, em vez de olhar um único passo, vamos repeti-lo e observar a saída "
            "convergir para o alvo, iteração após iteração.",
            is_checkpoint=True,
        )
        return pipeline + [bridge] + convergence[1:]
