"""Demonstração — Forward e backward tradicionais, até convergir.

Duas partes, um único fio condutor: primeiro o mecanismo (o que o forward
calcula — incluindo a ativação sigmoide, que é o que torna o neurônio não
linear — e o que a regra da cadeia calcula no backward, agora com três elos
em vez de dois); depois o resultado de repetir esse mesmo passo várias
vezes seguidas — um neurônio de verdade convergindo para um alvo, número
por número, até a saída ficar arbitrariamente perto do que se queria.

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

from efficient_nn_lab.backprop.activation import sigmoid, sigmoid_derivative
from efficient_nn_lab.bitnet.linear import loss_gradient_wrt_y, squared_error_loss
from efficient_nn_lab.bitnet.ste import sgd_update
from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.core.math_utils import ease_in_out, lerp

#: Interior sliding frames generated per gradient-descent iteration in the
#: convergence chart — enough for the point's motion toward its new value
#: to read as a clear slide, not a jump.
_SUBSTEPS_PER_ITERATION = 10

#: Fixed z-window the sigmoid curve is drawn over in both phases — wide
#: enough to show both flat/saturated ends, so "where on the curve am I"
#: is always legible, not just the steep middle.
_Z_RANGE = (-6.0, 6.0)

#: |y - target| below this counts as "close enough" -- both phases repeat
#: their cycle until the output crosses this line, instead of stopping
#: after a fixed, possibly-too-early or wastefully-late iteration count.
_CONVERGENCE_EPS = 0.05

#: Safety cap on repetitions regardless of convergence, so a slow learning
#: rate combined with a hard-to-reach target (both slider-adjustable)
#: cannot generate an unbounded number of steps.
_MAX_ITERATIONS = 25


def _activation_fields(z: float, target: float) -> dict[str, float]:
    """Everything the sigmoid inset (point + tangent + descent arrow) needs."""
    y = float(sigmoid(z))
    slope = float(sigmoid_derivative(z))
    grad_y = float(loss_gradient_wrt_y(y, target))
    grad_z = grad_y * slope
    return {"z": z, "y": y, "slope": slope, "grad_y": grad_y, "grad_z": grad_z}


class TraditionalBackpropDemo(DemoModule):
    title = "Backprop -> Forward e backward classicos"
    slug = "backprop.classic"
    description = (
        "Como o forward (combinação linear + ativação sigmoide) e o backward (regra da "
        "cadeia com três elos) funcionam — e o que acontece quando se repete esse passo "
        "várias vezes: a saída converge para o alvo."
    )

    def __init__(self) -> None:
        self.x = 2.0
        self.target = 0.9
        self.w_init = -1.0
        self.learning_rate = 3.0
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "target": {"label": "Alvo (0-1)", "min": 0.1, "max": 0.95, "step": 0.05, "value": self.target},
            "learning_rate": {"label": "Taxa de aprendizado", "min": 0.5, "max": 5.0, "step": 0.5, "value": self.learning_rate},
        }

    # -- part 1: the mechanism, repeated until the output is close enough --
    def _build_one_pipeline_cycle(self, iteration: int, w: float, is_last_cycle: bool) -> list[Frame]:
        """The same 9-step forward/backward walkthrough, for one iteration.

        Labels are prefixed with the iteration number so re-running this
        cycle for iteration 2, 3, ... reads as "the same steps again, from
        an updated w" rather than a confusing repeat of identical labels.
        """
        x = self.x
        z = w * x
        act = _activation_fields(z, self.target)
        y, slope, grad_y, grad_z = act["y"], act["slope"], act["grad_y"], act["grad_z"]
        diff = y - self.target
        loss = squared_error_loss(y, self.target)
        grad_w = grad_z * x
        w_updated = sgd_update(w, grad_w, self.learning_rate)
        converged = abs(diff) < _CONVERGENCE_EPS

        base = {
            "kind": "backprop_pipeline",
            "x": x, "w": w, "z": z, "y": y, "target": self.target, "diff": diff, "loss": loss,
            "slope": slope, "grad_y": grad_y, "grad_z": grad_z, "grad_w": grad_w,
            "w_updated": w_updated, "lr": self.learning_rate, "iteration": iteration,
            "z_reveal": 0.0, "y_reveal": 0.0, "target_reveal": 0.0, "diff_reveal": 0.0, "loss_reveal": 0.0,
            "grady_reveal": 0.0, "gradz_reveal": 0.0, "gradw_reveal": 0.0, "update_reveal": 0.0, "w_pulse": 0.0,
        }
        prefix = f"Iteração {iteration} — "

        def frame(label: str, explanation: str, equation: str = "", **overrides) -> Frame:
            values = dict(base)
            values.update(overrides)
            return Frame(prefix + label, values, explanation, equation)

        if iteration == 1:
            intro = (
                "Um neurônio com ativação sigmoide: primeiro combina w e x linearmente, "
                "depois espreme o resultado em (0, 1) com sigma. Primeiro o forward "
                "calcula y; depois o backward calcula o quanto mudar w."
            )
        else:
            intro = (
                f"O mesmo neurônio, o mesmo ciclo de 9 passos — mas agora começando do "
                f"w = {w:g} que a iteração anterior deixou. É exatamente essa repetição, "
                "iteração após iteração, que faz y se aproximar do alvo."
            )

        if is_last_cycle and converged:
            update_explanation = (
                f"O peso anda um pequeno passo no sentido contrário ao gradiente: w vira {w_updated:g}. "
                f"Agora a saída ({y:g}) já está perto o suficiente do alvo ({self.target:g}, diferença "
                f"de {abs(diff):.3f}) — o ciclo para de se repetir aqui."
            )
        elif is_last_cycle:
            update_explanation = (
                f"O peso anda um pequeno passo no sentido contrário ao gradiente: w vira {w_updated:g}. "
                f"A diferença ainda é {abs(diff):.3f}, mas chegamos ao limite de iterações mostradas "
                "neste passo a passo detalhado — a próxima parte continua daqui, de forma resumida."
            )
        else:
            update_explanation = (
                f"O peso anda um pequeno passo no sentido contrário ao gradiente: w vira {w_updated:g}, um "
                f"pouco mais perto do valor que aproximaria y do alvo. A saída ainda está a {abs(diff):.3f} "
                f"do alvo — longe o bastante para o ciclo se repetir: a iteração {iteration + 1} refaz "
                "exatamente os mesmos 9 passos, agora partindo desse w atualizado."
            )

        return [
            frame("O neurônio", intro, equation="y = sigma(w * x)"),
            frame(
                "Forward, parte 1: combinação linear",
                f"Antes da ativação, w e x só se multiplicam: z = w * x = {w:g} * {x:g} = {z:g}.",
                equation="z = w * x",
                z_reveal=1.0,
            ),
            frame(
                "Forward, parte 2: ativação sigmoide",
                f"z passa pela sigmoide, que o espreme para dentro de (0, 1): "
                f"y = sigma({z:g}) = {y:g}. É este o ponto marcado na curva ao lado.",
                equation="y = sigma(z) = 1 / (1 + e^-z)",
                z_reveal=1.0, y_reveal=1.0,
            ),
            frame(
                "Comparar com o alvo",
                f"O alvo é {self.target:g}; a saída atual é {y:g} — {'ainda longe' if not converged else 'já perto'}. "
                "Essa diferença é o que o treino tenta reduzir a cada passo.",
                z_reveal=1.0, y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0,
            ),
            frame(
                "A perda (loss)",
                "A perda resume a diferença em um único número, sempre positivo, que cresce "
                "quanto mais longe do alvo a saída estiver.",
                equation="L = 1/2 (y - target)^2",
                z_reveal=1.0, y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0,
            ),
            frame(
                "Backward, primeiro elo: dL/dy",
                f"O backward começa no fim: o quanto a perda muda se y mudasse um "
                f"pouco. dL/dy = y - target = {grad_y:g}.",
                equation="dL/dy = y - target",
                z_reveal=1.0, y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0, grady_reveal=1.0,
            ),
            frame(
                "Backward, segundo elo: atravessando a sigmoide",
                f"A sigmoide também tem derivada: sigma'(z) = y*(1-y) = {slope:g}. "
                f"dL/dz = dL/dy * sigma'(z) = {grad_z:g} — é essa inclinação que a reta "
                "tangente ao lado mostra, e o sinal dela diz para que lado mover z.",
                equation="dL/dz = dL/dy * sigma'(z)",
                z_reveal=1.0, y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0,
                grady_reveal=1.0, gradz_reveal=1.0,
            ),
            frame(
                "Backward, terceiro elo: dL/dw",
                f"Como z = w * x, o último elo da cadeia é multiplicar por x: "
                f"dL/dw = dL/dz * x = {grad_w:g}.",
                equation="dL/dw = dL/dz * dz/dw = dL/dz * x",
                z_reveal=1.0, y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0,
                grady_reveal=1.0, gradz_reveal=1.0, gradw_reveal=1.0,
            ),
            frame(
                "Atualizar o peso",
                update_explanation,
                equation="w <- w - taxa * dL/dw",
                z_reveal=1.0, y_reveal=1.0, target_reveal=1.0, diff_reveal=1.0, loss_reveal=1.0,
                grady_reveal=1.0, gradz_reveal=1.0, gradw_reveal=1.0, update_reveal=1.0, w_pulse=1.0,
            ),
        ]

    def _build_pipeline_checkpoints(self) -> list[Frame]:
        frames: list[Frame] = []
        w = self.w_init
        iteration = 1
        while True:
            z = w * self.x
            y = float(sigmoid(z))
            converged = abs(y - self.target) < _CONVERGENCE_EPS
            is_last_cycle = converged or iteration >= _MAX_ITERATIONS
            cycle = self._build_one_pipeline_cycle(iteration, w, is_last_cycle)
            frames.extend(cycle)
            if is_last_cycle:
                break
            w = float(cycle[-1].values["w_updated"])
            iteration += 1
        return frames

    # -- part 2: the same step, repeated until close enough (or capped) -----
    def _run_gradient_descent(self) -> dict[str, list[float]]:
        """Repeats forward+backward+update until |y - target| < eps.

        Stops as soon as convergence is reached (fewer wasted iterations
        for an easy target/learning-rate combo) or after `_MAX_ITERATIONS`
        (so a slow one can't run forever) — never a fixed count.
        """
        w = self.w_init
        hist: dict[str, list[float]] = {"w": [], "z": [], "y": [], "loss": []}
        for _ in range(_MAX_ITERATIONS + 1):
            z = w * self.x
            act = _activation_fields(z, self.target)
            y = act["y"]
            loss = squared_error_loss(y, self.target)
            hist["w"].append(w)
            hist["z"].append(z)
            hist["y"].append(y)
            hist["loss"].append(loss)
            if abs(y - self.target) < _CONVERGENCE_EPS:
                break
            grad_w = act["grad_z"] * self.x
            w = sgd_update(w, grad_w, self.learning_rate)
        return hist

    def _build_convergence_frames(self) -> list[Frame]:
        h = self._run_gradient_descent()
        w_hist, z_hist, y_hist, loss_hist = h["w"], h["z"], h["y"], h["loss"]
        n = len(w_hist) - 1
        y_min = min(0.0, min(y_hist), self.target) - 0.1
        y_max = max(max(y_hist), self.target) + 0.1
        loss_min = max(1e-5, min(loss_hist) * 0.5)
        loss_max = max(loss_hist) * 1.5

        def chart_values(k_exact: int, w_val: float, z_val: float, loss_val: float) -> dict[str, object]:
            # y is derived from z (not interpolated independently) so the
            # point sliding along the sigmoid inset and the point sliding
            # along the top line chart are always the same physical value.
            act = _activation_fields(z_val, self.target)
            y_val = act["y"]
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
                # grows by one point per iteration, same as w/y/loss above --
                # the sigmoid-curve panel must leave a trail of where the
                # point has already been, not just relocate a single dot.
                "z_trail": np.array(z_hist[:k_exact] + [z_val]),
                "z": act["z"], "slope": act["slope"], "grad_y": act["grad_y"], "grad_z": act["grad_z"],
            }

        initial_diff = abs(y_hist[0] - self.target)
        if n == 0:
            initial_explanation = (
                f"Peso inicial w = {w_hist[0]:g} já produz y = {y_hist[0]:.3f}, perto o suficiente do "
                f"alvo {self.target:g} (diferença de {initial_diff:.3f}) — nenhuma iteração de treino "
                "é necessária desta vez."
            )
        else:
            initial_explanation = (
                f"Peso inicial w = {w_hist[0]:g}, a {initial_diff:.3f} do que faria y bater com o "
                f"alvo {self.target:g}. Cada iteração daqui pra frente repete exatamente o "
                "forward + backward + atualização da parte anterior, até chegar perto o suficiente."
            )
        frames: list[Frame] = [
            Frame(
                "Antes de treinar (iteração 0)",
                chart_values(0, w_hist[0], z_hist[0], loss_hist[0]),
                initial_explanation,
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
                            lerp(z_hist[k - 1], z_hist[k], t),
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
                if diff < _CONVERGENCE_EPS:
                    explanation += " A saída já está perto o suficiente do alvo — é isso que 'convergir' significa aqui."
                else:
                    explanation += " Chegamos ao limite de iterações mostradas, mesmo sem convergência total."
            frames.append(
                Frame(
                    f"Iteração {k}",
                    chart_values(k, w_hist[k], z_hist[k], loss_hist[k]),
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
