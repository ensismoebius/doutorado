"""Demonstração SNN — Codificação Poisson de uma imagem, 30 passos.

Single question answered: what does rate coding look like on something
students actually recognize, instead of an abstract 1-D waveform? Same
rule as `poisson_coding.py` (`snn/encoding.py`'s `spike_probability`),
applied per-pixel to a real photo: every pixel is its own independent
Poisson-coded neuron, firing with probability proportional to its own
brightness. At any single time-step the result looks like noise; only
the 30 time-steps *together* let the eye reconstruct the picture, since
brighter pixels reliably spike more often than dark ones.

Every one of the 30 steps is its own checkpoint (no interior tweens) —
each step is a fresh, independent coin flip per pixel, not a smooth
continuation of the previous one, so there is nothing meaningful to
interpolate between two consecutive steps (mirrors guided_sequence.py's
fixed-checkpoint-sequence pattern, not spike_generation.py's continuous
sweep).
"""

from __future__ import annotations

from pathlib import Path

from efficient_nn_lab.core.demo import DemoModule, Frame
from efficient_nn_lab.snn.encoding import load_grayscale_image, poisson_spike_frames

_IMAGE_PATH = Path(__file__).resolve().parents[2] / "resources" / "images" / "patrick.jpg"
#: (rows, cols) -- kept close to the source photo's 16:9 aspect ratio, and
#: high enough resolution that Patrick is still recognizable at a glance
#: (part of the point, per the lecturer: this is a fun, memorable example),
#: not just a correct-but-abstract gray blob.
_IMAGE_SIZE = (108, 192)
_N_STEPS = 30


class PoissonImageCodingDemo(DemoModule):
    title = "SNN -> Codificação Poisson (imagem)"
    slug = "snn.poisson_image"
    description = (
        "Cada pixel de uma imagem real vira um neurônio: dispara, passo a passo, com probabilidade "
        "proporcional ao seu brilho. Só a soma de vários passos faz a imagem reaparecer."
    )

    def __init__(self) -> None:
        self.max_rate = 0.9
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {
            "max_rate": {
                "label": "Taxa máxima de disparo",
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
                "value": self.max_rate,
            }
        }

    def _build_frames(self) -> list[Frame]:
        image = load_grayscale_image(_IMAGE_PATH, _IMAGE_SIZE)
        spike_frames = poisson_spike_frames(image, _N_STEPS, self.max_rate)

        frames: list[Frame] = []
        for t in range(_N_STEPS):
            n_active = int(spike_frames[t].sum())
            frames.append(
                Frame(
                    label=f"t = {t}/{_N_STEPS - 1}",
                    values={
                        "kind": "poisson_image_coding",
                        "image": image,
                        "frame": spike_frames[t],
                        "t": t,
                        "n_total": _N_STEPS,
                        "max_rate": self.max_rate,
                    },
                    explanation=(
                        f"{n_active} de {image.size} pixels dispararam neste passo. Cada pixel sorteia "
                        "de forma independente: probabilidade = seu brilho x taxa máxima — os pixels "
                        "claros do rosto disparam bem mais que o fundo escuro, mas nunca com certeza."
                    ),
                    equation="P(pixel dispara em t) = brilho(pixel) . taxa_máxima",
                )
            )
        return frames
