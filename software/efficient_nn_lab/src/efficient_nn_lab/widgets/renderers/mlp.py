"""Renderer for `backprop.mlp` — the 3 -> 2 -> 2 -> 1 network, walked one
neuron at a time."""

from __future__ import annotations

from efficient_nn_lab.app.theme import (
    ACCENT_COLOR,
    BITNET_COLOR,
    CONVERGE_COLOR,
    NEUTRAL_COLOR,
    SNN_COLOR,
)
from efficient_nn_lab.backprop.activation import sigmoid_derivative


class MlpNetworkRendererMixin:
    """Draws the `mlp_network` frame kind."""

    # ================================================================
    # backprop/demos/multilayer_network.py — 3 -> 2 -> 2 -> 1, sigmoid
    # everywhere. Walked one neuron at a time (forward left-to-right, then
    # backward right-to-left); the "active" neuron gets a glow, its
    # incoming edges get weight labels, and the detail panel focuses on it.
    # Unlike the single-neuron demo, every neuron keeps its *own* small
    # sigmoid panel, all visible at once (not one shared/switching inset)
    # so students can watch each neuron's point/tangent/gradient evolve
    # independently as the walkthrough proceeds.
    # ================================================================
    _MLP_X = [(0.6, 5.6), (0.6, 3.5), (0.6, 1.4)]
    _MLP_L1 = [(3.0, 4.7), (3.0, 2.3)]
    _MLP_L2 = [(5.4, 4.7), (5.4, 2.3)]
    _MLP_O = (7.8, 3.5)
    _MLP_TARGET = (7.8, 5.6)
    _MLP_LOSS = (7.8, 1.4)
    _MLP_NAMES = ["L1-A", "L1-B", "L2-C", "L2-D", "O"]
    _MLP_POS = {"L1-A": _MLP_L1[0], "L1-B": _MLP_L1[1], "L2-C": _MLP_L2[0], "L2-D": _MLP_L2[1], "O": _MLP_O}
    _MLP_INPUT_POS = {
        "L1-A": _MLP_X, "L1-B": _MLP_X,
        "L2-C": _MLP_L1, "L2-D": _MLP_L1,
        "O": _MLP_L2,
    }
    # figure-fraction rects for the 5 per-neuron sigmoid panels, arranged
    # in the same left-to-right layer order as the diagram itself (L1
    # column, L2 column, output column) so "which graph is which neuron"
    # is obvious without reading labels.
    _MLP_INSET_RECTS = {
        "L1-A": (0.55, 0.56, 0.13, 0.38),
        "L1-B": (0.55, 0.08, 0.13, 0.38),
        "L2-C": (0.705, 0.56, 0.13, 0.38),
        "L2-D": (0.705, 0.08, 0.13, 0.38),
        "O": (0.86, 0.32, 0.13, 0.38),
    }

    def _render_mlp_network(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(-0.1, 8.6), ylim=(-3.0, 6.6))
        x, target = values["x"], float(values["target"])
        w1, w2, w3 = values["w1"], values["w2"], values["w3"]
        z1, y1, z2, y2, zO, yO = values["z1"], values["y1"], values["z2"], values["y2"], float(values["zO"]), float(values["yO"])
        gz1, gz2, gzO = values["grad_z1"], values["grad_z2"], float(values["grad_zO"])
        fwd = {
            "L1-A": float(values["fwd_l1a"]), "L1-B": float(values["fwd_l1b"]),
            "L2-C": float(values["fwd_l2c"]), "L2-D": float(values["fwd_l2d"]), "O": float(values["fwd_o"]),
        }
        bwd = {
            "L1-A": float(values["bwd_l1a"]), "L1-B": float(values["bwd_l1b"]),
            "L2-C": float(values["bwd_l2c"]), "L2-D": float(values["bwd_l2d"]), "O": float(values["bwd_o"]),
        }
        y_val = {"L1-A": float(y1[0]), "L1-B": float(y1[1]), "L2-C": float(y2[0]), "L2-D": float(y2[1]), "O": yO}
        gz_val = {"L1-A": float(gz1[0]), "L1-B": float(gz1[1]), "L2-C": float(gz2[0]), "L2-D": float(gz2[1]), "O": gzO}
        w_row = {"L1-A": w1[0], "L1-B": w1[1], "L2-C": w2[0], "L2-D": w2[1], "O": w3[0]}
        loss_reveal = float(values["loss_reveal"])
        update_reveal = float(values["update_reveal"])
        active = values.get("active", "")

        # full skeleton: every box and every weighted edge, from frame one.
        for p in self._MLP_X + self._MLP_L1 + self._MLP_L2 + [self._MLP_O, self._MLP_TARGET, self._MLP_LOSS]:
            self._skeleton_box(*p, w=1.4, h=0.75)
        for name in self._MLP_NAMES:
            for p_in in self._MLP_INPUT_POS[name]:
                self._skeleton_arrow(p_in, self._MLP_POS[name])
        self._skeleton_arrow(self._MLP_O, self._MLP_TARGET)
        self._skeleton_arrow(self._MLP_O, self._MLP_LOSS)

        for i, p in enumerate(self._MLP_X):
            self._box(*p, f"x{i+1} = {x[i]:g}", NEUTRAL_COLOR, w=1.4, h=0.75, fontsize=9)

        for name in self._MLP_NAMES:
            r = fwd[name]
            if r <= 0.02:
                continue
            pos = self._MLP_POS[name]
            glow = 1.0 if active == name else 0.0
            for p_in in self._MLP_INPUT_POS[name]:
                self._flow_arrow(p_in, pos, r, ACCENT_COLOR if glow else BITNET_COLOR)
            text = f"{name}\ny = {y_val[name]:.2f}"
            if bwd[name] > 0.02:
                text += f"\n∂L/∂z={gz_val[name]:.3f}"
            self._box(*pos, text, CONVERGE_COLOR if bwd[name] > 0.02 else BITNET_COLOR, alpha=r, w=1.4, h=0.75, glow=glow, fontsize=8)
            if glow:
                self._equation_near(*pos, "y = σ(Σ w·x)", r, box_h=0.75)

        self._flow_arrow(self._MLP_O, self._MLP_TARGET, loss_reveal, NEUTRAL_COLOR)
        self._box(*self._MLP_TARGET, f"alvo = {target:g}", NEUTRAL_COLOR, alpha=loss_reveal, w=1.4, h=0.75, fontsize=9)
        self._flow_arrow(self._MLP_O, self._MLP_LOSS, loss_reveal, SNN_COLOR)
        self._box(*self._MLP_LOSS, f"loss = {float(values['loss']):.3f}", SNN_COLOR, alpha=loss_reveal, w=1.4, h=0.75, fontsize=9)

        # backward: a thin orange return-edge drawn *behind* each active
        # neuron's incoming connections, showing gradient flowing the
        # opposite way along the same wires.
        for name in self._MLP_NAMES:
            if bwd[name] <= 0.02:
                continue
            for p_in in self._MLP_INPUT_POS[name]:
                self._flow_arrow(self._MLP_POS[name], p_in, bwd[name] * 0.6, SNN_COLOR)

        detail = str(values.get("active_detail", ""))
        if detail:
            self._ax.text(
                0.1, -1.1, detail, ha="left", va="top", fontsize=8.5, color="black",
                family="monospace", linespacing=1.6,
            )

        if update_reveal > 0.02:
            self._fading_text(4.2, -2.3, "todos os pesos atualizados: w ← w - taxa · ∂L/∂w", ACCENT_COLOR, update_reveal, fontsize=9)

        z_val = {"L1-A": float(z1[0]), "L1-B": float(z1[1]), "L2-C": float(z2[0]), "L2-D": float(z2[1]), "O": zO}
        slope_val = {name: float(sigmoid_derivative(z_val[name])) for name in self._MLP_NAMES}
        for name in self._MLP_NAMES:
            ax = self._get_inset(name, self._MLP_INSET_RECTS[name])
            self._paint_sigmoid(
                ax, z=z_val[name], y=y_val[name], slope=slope_val[name], grad_z=gz_val[name],
                point_reveal=fwd[name], tangent_reveal=fwd[name], arrow_reveal=bwd[name],
                title=name, compact=True,
            )

