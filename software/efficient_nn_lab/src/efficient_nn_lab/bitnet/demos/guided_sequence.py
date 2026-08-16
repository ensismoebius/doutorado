""""Do peso real ao BitNet" — sequência didática obrigatória.

(ESPECIFICACAO_DLVL.md #30, with the "peso oculto" framing from #15.)

Fixed numbers, not user-configurable: this is the one walkthrough every
first-time viewer should see end to end, exactly as specified. One
persistent diagram accumulates every box (w, Q(w), x, y, target, loss,
gradient) in fixed positions across all 10 steps — nothing is ever wiped;
later steps just reveal more of the same picture, and the weight box's
displayed number visibly morphs from 0.80 to 0.84 at the update step
instead of jumping. The central message is the last frame: the continuous
parameter changed, but the representation actually used in the forward
pass can stay the same (+1) — the shadow weight is real, even when
invisible in the forward pass.
"""

from __future__ import annotations

from efficient_nn_lab.core.demo import DemoModule, Frame, build_sequence
from efficient_nn_lab.bitnet.linear import (
    loss_gradient_wrt_y,
    quantized_forward,
    squared_error_loss,
)
from efficient_nn_lab.bitnet.quantization import DEFAULT_THRESHOLD, ternary_quantize
from efficient_nn_lab.bitnet.ste import sgd_update, ste_backward

_W0 = 0.80
_X = 2.0
_TARGET = 4.0
_LEARNING_RATE = 0.01
_TWEEN_STEPS = 16


class GuidedBitNetDemo(DemoModule):
    title = "BitNet -> Exemplo guiado"
    slug = "bitnet.guided"
    description = 'Sequência fixa "Do peso real ao BitNet": um ciclo completo de forward, loss, STE e atualização.'

    def _build_frames(self) -> list[Frame]:
        w = _W0
        q = ternary_quantize(w, DEFAULT_THRESHOLD)
        result = quantized_forward((_X,), (w,), DEFAULT_THRESHOLD)
        y = result.y
        loss = squared_error_loss(y, _TARGET)
        grad_y = loss_gradient_wrt_y(y, _TARGET)
        grad_w = ste_backward(grad_y * _X)
        w_new = sgd_update(w, grad_w, _LEARNING_RATE)
        q_new = ternary_quantize(w_new, DEFAULT_THRESHOLD)

        base = {
            "kind": "guided_pipeline",
            "w_value": w,
            "q_value": q,
            "q_reveal": 0.0,
            "q_pulse": 0.0,
            "x_reveal": 0.0,
            "y_value": y,
            "y_reveal": 0.0,
            "target_value": _TARGET,
            "target_reveal": 0.0,
            "loss_value": loss,
            "loss_reveal": 0.0,
            "grad_value": grad_w,
            "grad_reveal": 0.0,
            "ste_reveal": 0.0,
            "update_reveal": 0.0,
            "step_number": 1,
        }

        def frame(label: str, explanation: str, equation: str = "", **overrides) -> Frame:
            values = dict(base)
            values.update(overrides)
            return Frame(label, values, explanation, equation)

        checkpoints = [
            frame("Passo 1 — peso real", f"w = {w:.2f}.", step_number=1),
            frame(
                "Passo 2 — quantização",
                f"Q(w) = Q({w:.2f}) = {q:+d}.",
                equation="Q(w) = +1 se w > tau; -1 se w < -tau; 0 caso contrario.",
                q_reveal=1.0,
                step_number=2,
            ),
            frame("Passo 3 — entrada", f"x = {_X:g}.", q_reveal=1.0, x_reveal=1.0, step_number=3),
            frame(
                "Passo 4 — saída",
                f"y = x . Q(w) = {_X:g} . {q:+d} = {y:g}.",
                equation="y = x . Q(w)",
                q_reveal=1.0,
                x_reveal=1.0,
                y_reveal=1.0,
                step_number=4,
            ),
            frame(
                "Passo 5 — alvo",
                f"target = {_TARGET:g}.",
                q_reveal=1.0,
                x_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
                step_number=5,
            ),
            frame(
                "Passo 6 — perda",
                f"loss = 1/2 (y - target)^2 = 1/2 ({y:g} - {_TARGET:g})^2 = {loss:g}.",
                equation="L = 1/2 (y - target)^2",
                q_reveal=1.0,
                x_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
                loss_reveal=1.0,
                step_number=6,
            ),
            frame(
                "Passo 7 — gradiente",
                f"dL/dy = y - target = {grad_y:g}. Via STE, dL/dw ~= dL/dy . x = {grad_w:g}.",
                equation="dL/dw ~= (y - target) . x  [STE]",
                q_reveal=1.0,
                x_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
                loss_reveal=1.0,
                grad_reveal=1.0,
                step_number=7,
            ),
            frame(
                "Passo 8 — STE",
                "O STE deixa esse gradiente atravessar a quantização sem alteração, "
                "como se Q fosse a função identidade no backward.",
                q_reveal=1.0,
                x_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
                loss_reveal=1.0,
                grad_reveal=1.0,
                ste_reveal=1.0,
                step_number=8,
            ),
            frame(
                "Passo 9 — atualização",
                f"w <- w - lr . dL/dw = {w:.2f} - {_LEARNING_RATE:g} . ({grad_w:g}) = {w_new:.2f}.",
                equation="w <- w - eta . dL/dw",
                w_value=w_new,
                q_reveal=1.0,
                x_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
                loss_reveal=1.0,
                grad_reveal=1.0,
                ste_reveal=1.0,
                update_reveal=1.0,
                step_number=9,
            ),
            frame(
                "Passo 10 — nova quantização",
                (
                    f"Q({w_new:.2f}) = {q_new:+d}. "
                    + (
                        "A representação usada no forward não mudou, mesmo com o parâmetro real atualizado."
                        if q_new == q
                        else "Desta vez a representação usada no forward também mudou."
                    )
                ),
                w_value=w_new,
                q_value=q_new,
                q_reveal=1.0,
                q_pulse=1.0,
                x_reveal=1.0,
                y_reveal=1.0,
                target_reveal=1.0,
                loss_reveal=1.0,
                grad_reveal=1.0,
                ste_reveal=1.0,
                update_reveal=1.0,
                step_number=10,
            ),
        ]
        return build_sequence(checkpoints, steps=_TWEEN_STEPS)
