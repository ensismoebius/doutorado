"""Renderers for the block-diagram pipelines: classic backprop, the BitNet
forward pass, the STE's dual path, and the guided BitNet walkthrough."""

from __future__ import annotations

from efficient_nn_lab.app.theme import (
    ACCENT_COLOR,
    BITNET_COLOR,
    CONVERGE_COLOR,
    NEUTRAL_COLOR,
    SNN_COLOR,
)


class PipelineRenderersMixin:
    """Draws the `backprop_pipeline`, `forward_pipeline`, `ste_pipeline`
    and `guided_pipeline` frame kinds."""

    # ================================================================
    # backprop/demos/traditional_gd.py — one weight, sigmoid activation: the
    # forward (z = w*x, y = sigma(z)) and backward (three-link chain rule)
    # that BitNet's STE and the SNN's surrogate gradient are variations on.
    # A sigmoid-curve inset (point + tangent + descent direction) shares
    # the figure with the block diagram — see _paint_sigmoid.
    # ================================================================
    _BP_X = (0.7, 5.3)
    _BP_W = (0.7, 3.0)
    _BP_Z = (3.0, 4.15)
    _BP_Y = (5.6, 4.15)
    _BP_TARGET = (5.6, 6.15)
    _BP_LOSS = (5.6, 2.15)
    _BP_GRAD_Y = (3.0, 1.1)
    _BP_GRAD_Z = (5.6, 0.15)
    _BP_GRAD_W = (0.7, 1.1)

    def _render_backprop_pipeline(self, values: dict[str, object]) -> None:
        # Widened defensively past the box/equation-text extents below
        # (tests/test_widgets_no_clipping.py found no actual overflow, but
        # _BP_Y's box came within 0.05 units of xlim and _BP_GRAD_Z's
        # equation text already sits past the old ylim, only absorbed by
        # this axes' own figure-position padding) -- extra headroom here
        # costs nothing and removes the fragility.
        self._reset_axes(xlim=(-0.5, 6.9), ylim=(-0.7, 7.0))
        z_reveal = float(values["z_reveal"])
        y_reveal = float(values["y_reveal"])
        target_reveal = float(values["target_reveal"])
        diff_reveal = float(values["diff_reveal"])
        loss_reveal = float(values["loss_reveal"])
        grady_reveal = float(values["grady_reveal"])
        gradz_reveal = float(values["gradz_reveal"])
        gradw_reveal = float(values["gradw_reveal"])
        update_reveal = float(values["update_reveal"])
        w_pulse = float(values["w_pulse"])

        boxes = (
            self._BP_X, self._BP_W, self._BP_Z, self._BP_Y, self._BP_TARGET, self._BP_LOSS,
            self._BP_GRAD_Y, self._BP_GRAD_Z, self._BP_GRAD_W,
        )
        for p in boxes:
            self._skeleton_box(*p, w=1.5)
        self._skeleton_arrow(self._BP_X, self._BP_Z)
        self._skeleton_arrow(self._BP_W, self._BP_Z)
        self._skeleton_arrow(self._BP_Z, self._BP_Y)
        self._skeleton_arrow(self._BP_Y, self._BP_TARGET)
        self._skeleton_arrow(self._BP_Y, self._BP_LOSS)
        self._skeleton_arrow(self._BP_LOSS, self._BP_GRAD_Y)
        self._skeleton_arrow(self._BP_GRAD_Y, self._BP_GRAD_Z)
        self._skeleton_arrow(self._BP_GRAD_Z, self._BP_GRAD_W)
        self._skeleton_arrow(self._BP_GRAD_W, self._BP_W)

        self._box(*self._BP_X, f"x = {values['x']:g}", NEUTRAL_COLOR, w=1.5)
        self._box(*self._BP_W, f"w = {values['w']:g}", BITNET_COLOR, w=1.5, glow=w_pulse)

        self._flow_arrow(self._BP_X, self._BP_Z, z_reveal, NEUTRAL_COLOR)
        self._flow_arrow(self._BP_W, self._BP_Z, z_reveal, BITNET_COLOR)
        self._box(*self._BP_Z, f"z = {values['z']:g}", BITNET_COLOR, alpha=z_reveal, w=1.5, glow=float(values.get("z_glow", 0.0)))
        self._equation_near(*self._BP_Z, "z = w · x", z_reveal)

        self._flow_arrow(self._BP_Z, self._BP_Y, y_reveal, CONVERGE_COLOR)
        self._box(*self._BP_Y, f"y = σ(z)\n= {values['y']:.3f}", CONVERGE_COLOR, alpha=y_reveal, w=1.9, glow=float(values.get("y_glow", 0.0)))
        self._equation_near(*self._BP_Y, "y = σ(z) = 1/(1+e⁻ᶻ)", y_reveal)

        self._flow_arrow(self._BP_Y, self._BP_TARGET, target_reveal, NEUTRAL_COLOR)
        self._box(*self._BP_TARGET, f"target = {values['target']:g}", NEUTRAL_COLOR, alpha=target_reveal, w=1.5, glow=float(values.get("target_glow", 0.0)))
        self._fading_text(
            (self._BP_Y[0] + self._BP_TARGET[0]) / 2 - 1.05, (self._BP_Y[1] + self._BP_TARGET[1]) / 2,
            f"diferença = {values['diff']:.3f}", ACCENT_COLOR, diff_reveal, fontsize=8,
        )

        self._flow_arrow(self._BP_Y, self._BP_LOSS, loss_reveal, SNN_COLOR)
        self._box(*self._BP_LOSS, f"loss = {values['loss']:.3f}", SNN_COLOR, alpha=loss_reveal, w=1.5, glow=float(values.get("loss_glow", 0.0)))
        self._equation_near(*self._BP_LOSS, "L = ½ (y - target)²", loss_reveal)

        self._flow_arrow(self._BP_LOSS, self._BP_GRAD_Y, grady_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_Y, f"∂L/∂y = {values['grad_y']:.2f}", SNN_COLOR, alpha=grady_reveal, w=1.5, glow=float(values.get("grady_glow", 0.0)))
        self._equation_near(*self._BP_GRAD_Y, "∂L/∂y = y - target", grady_reveal)

        self._flow_arrow(self._BP_GRAD_Y, self._BP_GRAD_Z, gradz_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_Z, f"∂L/∂z = {values['grad_z']:.3f}", SNN_COLOR, alpha=gradz_reveal, w=1.6, glow=float(values.get("gradz_glow", 0.0)))
        self._equation_near(*self._BP_GRAD_Z, "∂L/∂z = ∂L/∂y · σ'(z)", gradz_reveal)

        self._flow_arrow(self._BP_GRAD_Z, self._BP_GRAD_W, gradw_reveal, SNN_COLOR)
        self._box(*self._BP_GRAD_W, f"∂L/∂w = {values['grad_w']:.2f}", SNN_COLOR, alpha=gradw_reveal, w=1.5, glow=float(values.get("gradw_glow", 0.0)))
        self._equation_near(*self._BP_GRAD_W, "∂L/∂w = ∂L/∂z · x", gradw_reveal)

        self._flow_arrow(self._BP_GRAD_W, self._BP_W, update_reveal, ACCENT_COLOR)
        self._fading_text(0.7, 2.0, f"atualizado -> {values['w_updated']:.3f}", ACCENT_COLOR, update_reveal, fontsize=8)
        self._equation_near(*self._BP_W, "w ← w - taxa · ∂L/∂w", update_reveal)

        ax = self._get_inset("main", (0.56, 0.1, 0.42, 0.85))
        # point_reveal/arrow_reveal are their own fields, deliberately
        # decoupled from y_reveal/gradz_reveal: those block-diagram reveals
        # reset every repeat cycle (see traditional_gd.py), but the point
        # sliding along this curve must stay continuous across cycles, or
        # the one thing meant to visibly show convergence would itself
        # blink out and back every time.
        inset_point_reveal = float(values.get("point_reveal", y_reveal))
        inset_arrow_reveal = float(values.get("arrow_reveal", gradz_reveal))
        self._paint_sigmoid(
            ax, z=float(values["z"]), y=float(values["y"]), slope=float(values["slope"]), grad_z=float(values["grad_z"]),
            point_reveal=inset_point_reveal, tangent_reveal=inset_point_reveal, arrow_reveal=inset_arrow_reveal,
            title="Ativação: onde estamos na curva",
        )

    # ================================================================
    # forward.py — inputs -> quantized weights -> sum -> y -> target -> loss
    # ================================================================
    _X1, _X2 = (1.0, 4.7), (1.0, 1.6)
    _W1, _W2 = (3.5, 4.7), (3.5, 1.6)
    _SUM = (6.3, 3.15)
    _Y = (8.5, 3.15)
    _TARGET = (8.5, 5.4)
    _LOSS = (8.5, 0.9)

    def _render_forward_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes()
        x1, x2 = values["x"]
        w1_real, w2_real = values["w_real"]
        w1q, w2q = values["w_quant"]
        threshold = float(values["threshold"])
        quant1_reveal = float(values["quant1_reveal"])
        quant2_reveal = float(values["quant2_reveal"])
        arrow1_fill = float(values["arrow1_fill"])
        arrow2_fill = float(values["arrow2_fill"])
        p1_reveal = float(values["product1_reveal"])
        p2_reveal = float(values["product2_reveal"])
        sum_reveal = float(values["sum_reveal"])
        y_reveal = float(values["y_reveal"])
        target_reveal = float(values["target_reveal"])
        diff_reveal = float(values["diff_reveal"])
        loss_reveal = float(values["loss_reveal"])
        h1, h2 = float(values["highlight1"]), float(values["highlight2"])

        # full skeleton, always visible, so the whole pipeline shape is
        # legible from the very first frame
        for p in (self._X1, self._X2, self._W1, self._W2, self._SUM, self._Y, self._TARGET, self._LOSS):
            self._skeleton_box(*p)
        self._skeleton_arrow(self._W1, self._SUM)
        self._skeleton_arrow(self._W2, self._SUM)
        self._skeleton_arrow(self._SUM, self._Y)
        self._skeleton_arrow(self._Y, self._TARGET)
        self._skeleton_arrow(self._Y, self._LOSS)

        self._box(*self._X1, f"x1 = {x1:g}", NEUTRAL_COLOR)
        self._box(*self._X2, f"x2 = {x2:g}", NEUTRAL_COLOR)

        w_label1 = f"Q(w1) = {round(w1q):+d}" if quant1_reveal >= 0.999 else f"w1 = {w1_real:.2f}"
        w_label2 = f"Q(w2) = {round(w2q):+d}" if quant2_reveal >= 0.999 else f"w2 = {w2_real:.2f}"
        self._box(*self._W1, w_label1, BITNET_COLOR if quant1_reveal > 0.5 else NEUTRAL_COLOR, glow=h1)
        self._box(*self._W2, w_label2, BITNET_COLOR if quant2_reveal > 0.5 else NEUTRAL_COLOR, glow=h2)
        self._equation_near(*self._W1, "Q(w) ∈ {+1, 0, -1}", max(quant1_reveal, quant2_reveal))

        self._flow_arrow(self._W1, self._SUM, arrow1_fill, ACCENT_COLOR if h1 > 0.3 else BITNET_COLOR)
        self._flow_arrow(self._W2, self._SUM, arrow2_fill, ACCENT_COLOR if h2 > 0.3 else BITNET_COLOR)
        self._fading_text(4.9, 4.35, f"produto = {values['product1']:g}", ACCENT_COLOR, p1_reveal, fontsize=9)
        self._fading_text(4.9, 1.95, f"produto = {values['product2']:g}", ACCENT_COLOR, p2_reveal, fontsize=9)

        self._box(*self._SUM, "soma", CONVERGE_COLOR, alpha=max(sum_reveal, 0.35 if sum_reveal else 0.0) or 0.01)
        self._fading_text(*self._SUM, "Σ", CONVERGE_COLOR, 1.0 if sum_reveal < 0.02 else 0.0, fontsize=14, weight="bold")
        self._equation_near(*self._SUM, "y = Σ xᵢ·Q(wᵢ)", sum_reveal)

        self._flow_arrow(self._SUM, self._Y, y_reveal, CONVERGE_COLOR)
        self._box(*self._Y, f"y = {values['y']:g}", CONVERGE_COLOR, alpha=y_reveal)

        self._flow_arrow(self._Y, self._TARGET, target_reveal, NEUTRAL_COLOR)
        self._box(*self._TARGET, f"target = {values['target']:g}", NEUTRAL_COLOR, alpha=target_reveal)
        self._fading_text(
            (self._Y[0] + self._TARGET[0]) / 2 + 0.6,
            (self._Y[1] + self._TARGET[1]) / 2,
            f"diferença = {values['diff']:g}",
            ACCENT_COLOR,
            diff_reveal,
            fontsize=9,
        )

        self._flow_arrow(self._Y, self._LOSS, loss_reveal, SNN_COLOR)
        self._box(*self._LOSS, f"loss = {values['loss']:g}", SNN_COLOR, alpha=loss_reveal)
        self._equation_near(*self._LOSS, "L = ½ (y - target)²", loss_reveal)

        ax = self._get_inset("forward_numberline", (0.56, 0.12, 0.42, 0.8))
        self._paint_weight_numberline(ax, w1_real, w2_real, w1q, w2q, threshold, quant1_reveal, quant2_reveal)

    # ================================================================
    # backward.py — forward path (always) + backward/STE path (reveals)
    # ================================================================
    _FWD_POSITIONS = [(1.2, 4.3), (3.8, 4.3), (6.4, 4.3), (9.0, 4.3)]
    _FWD_LABELS = ["peso real", "quantização", "peso ternário", "operação"]
    _BWD_POSITIONS = [(9.0, 1.6), (6.4, 1.6), (3.8, 1.6), (1.2, 1.6)]
    _BWD_LABELS = ["loss", "gradiente", "STE", "peso real"]

    def _render_ste_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(ylim=(0, 6))
        fwd = float(values["fwd_reveal"])
        bwd = float(values["bwd_reveal"])
        joined = float(values["joined_reveal"])
        # The one worked example every box references (see backward.py) --
        # carried through every frame so the diagram shows the same numbers
        # as the explanation text. Defaults keep old checkpoints renderable.
        w = float(values.get("w", 0.65))
        wq = int(values.get("w_quant", 1))
        x = float(values.get("x", 2.0))
        y = float(values.get("y", 2.0))
        loss = float(values.get("loss", 2.0))
        upstream = float(values.get("upstream_grad", -2.0))
        dq_real = float(values.get("dq_dw_real", 0.0))
        dq_ste = float(values.get("dq_dw_ste", 1.0))
        dl_real = float(values.get("dl_dw_real", 0.0))
        dl_ste = float(values.get("dl_dw_ste", upstream))
        tau = float(values.get("threshold", 0.5))

        fwd_labels = [
            f"peso real\nw = {w:.2f}",
            "quantização\nQ(w)",
            f"peso ternário\nQ(w) = {wq:+d}",
            f"operação\ny = {x:g}·Q(w) = {y:g}",
        ]
        bwd_labels = [
            f"loss\nL = {loss:g}",
            f"gradiente\n∂L/∂Q(w) = {upstream:g}",
            f"STE\ndQ/dw := {dq_ste:g}",
            f"peso real\n∂L/∂w = {dl_ste:g}",
        ]

        for p in self._FWD_POSITIONS + self._BWD_POSITIONS:
            self._skeleton_box(*p, w=2.0)
        for a, b in zip(self._FWD_POSITIONS, self._FWD_POSITIONS[1:]):
            self._skeleton_arrow((a[0] + 1.0, a[1]), (b[0] - 1.0, b[1]))
        for a, b in zip(self._BWD_POSITIONS, self._BWD_POSITIONS[1:]):
            self._skeleton_arrow((a[0] - 1.0, a[1]), (b[0] + 1.0, b[1]))

        for pos, label in zip(self._FWD_POSITIONS, fwd_labels):
            self._box(*pos, label, BITNET_COLOR, alpha=fwd, w=2.0, fontsize=9)
        for a, b in zip(self._FWD_POSITIONS, self._FWD_POSITIONS[1:]):
            self._flow_arrow((a[0] + 1.0, a[1]), (b[0] - 1.0, b[1]), fwd, BITNET_COLOR)
        self._fading_text(5.1, 5.35, "FORWARD", BITNET_COLOR, fwd, fontsize=12, weight="bold")
        self._fading_text(
            5.1, 5.02, f"y = {x:g}·Q(w), com Q({w:.2f}) = {wq:+d} (τ = {tau:g})",
            BITNET_COLOR, fwd, fontsize=8,
        )

        for pos, label in zip(self._BWD_POSITIONS, bwd_labels):
            self._box(*pos, label, SNN_COLOR, alpha=bwd, w=2.0, fontsize=9)
        for a, b in zip(self._BWD_POSITIONS, self._BWD_POSITIONS[1:]):
            self._flow_arrow((a[0] - 1.0, a[1]), (b[0] + 1.0, b[1]), bwd, SNN_COLOR)
        self._fading_text(5.1, 0.65, "BACKWARD (STE)", SNN_COLOR, bwd, fontsize=12, weight="bold")
        self._fading_text(
            5.1, 0.98, f"∂L/∂w = {dl_ste:g}  (dQ/dw: {dq_real:g} -> {dq_ste:g})",
            SNN_COLOR, bwd, fontsize=8,
        )

        # the two paths meet at "peso real": a highlighted ring makes the
        # loop visible once both are on screen. The box re-drawn on top
        # uses the forward label (its real value) since that is the one
        # the reader last saw there.
        self._box(*self._FWD_POSITIONS[0], fwd_labels[0], BITNET_COLOR, alpha=fwd, w=2.0, glow=joined, fontsize=9)
        self._fading_text(5.1, 3.0, "mesma quantização,\ncaminhos diferentes", ACCENT_COLOR, joined, fontsize=9)

    # ================================================================
    # guided_sequence.py — one growing computational graph, 10 steps
    # ================================================================
    _W_POS = (1.2, 3.1)
    _Q_POS = (3.5, 3.1)
    _X_POS = (3.5, 5.2)
    _Y_POS_G = (5.9, 4.15)
    _TARGET_POS_G = (8.3, 5.2)
    _LOSS_POS_G = (8.3, 3.1)
    _GRAD_POS = (5.9, 1.1)

    def _render_guided_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(ylim=(0, 6.4))
        q_reveal = float(values["q_reveal"])
        x_reveal = float(values["x_reveal"])
        y_reveal = float(values["y_reveal"])
        target_reveal = float(values["target_reveal"])
        loss_reveal = float(values["loss_reveal"])
        grad_reveal = float(values["grad_reveal"])
        ste_reveal = float(values["ste_reveal"])
        update_reveal = float(values["update_reveal"])
        q_pulse = float(values["q_pulse"])

        for p in (self._W_POS, self._Q_POS, self._X_POS, self._Y_POS_G, self._TARGET_POS_G, self._LOSS_POS_G, self._GRAD_POS):
            self._skeleton_box(*p)
        self._skeleton_arrow(self._W_POS, self._Q_POS)
        self._skeleton_arrow(self._Q_POS, self._Y_POS_G)
        self._skeleton_arrow(self._X_POS, self._Y_POS_G)
        self._skeleton_arrow(self._Y_POS_G, self._TARGET_POS_G)
        self._skeleton_arrow(self._Y_POS_G, self._LOSS_POS_G)
        self._skeleton_arrow(self._LOSS_POS_G, self._GRAD_POS)
        self._skeleton_arrow(self._GRAD_POS, self._W_POS)

        self._box(*self._W_POS, f"w = {values['w_value']:.2f}", BITNET_COLOR)
        self._flow_arrow(self._W_POS, self._Q_POS, q_reveal, BITNET_COLOR)
        self._box(*self._Q_POS, f"Q(w) = {round(values['q_value']):+d}", BITNET_COLOR, alpha=q_reveal, glow=q_pulse)
        self._equation_near(*self._Q_POS, "Q(w) = ±1/0 (limiar τ)", q_reveal)

        self._box(*self._X_POS, f"x = {2.0:g}", NEUTRAL_COLOR, alpha=x_reveal)
        self._flow_arrow(self._Q_POS, self._Y_POS_G, y_reveal, CONVERGE_COLOR)
        self._flow_arrow(self._X_POS, self._Y_POS_G, y_reveal, NEUTRAL_COLOR)
        self._box(*self._Y_POS_G, f"y = {values['y_value']:g}", CONVERGE_COLOR, alpha=y_reveal)
        self._equation_near(*self._Y_POS_G, "y = x · Q(w)", y_reveal)

        self._flow_arrow(self._Y_POS_G, self._TARGET_POS_G, target_reveal, NEUTRAL_COLOR)
        self._box(*self._TARGET_POS_G, f"target = {values['target_value']:g}", NEUTRAL_COLOR, alpha=target_reveal)

        self._flow_arrow(self._Y_POS_G, self._LOSS_POS_G, loss_reveal, SNN_COLOR)
        self._box(*self._LOSS_POS_G, f"loss = {values['loss_value']:g}", SNN_COLOR, alpha=loss_reveal)
        self._equation_near(*self._LOSS_POS_G, "L = ½ (y - target)²", loss_reveal)

        self._flow_arrow(self._LOSS_POS_G, self._GRAD_POS, grad_reveal, SNN_COLOR)
        self._box(*self._GRAD_POS, f"∂L/∂w ~= {values['grad_value']:g}", SNN_COLOR, alpha=grad_reveal)
        self._equation_near(*self._GRAD_POS, "∂L/∂w ≈ ∂L/∂y  (STE)", grad_reveal, side="above")

        self._flow_arrow(self._GRAD_POS, self._W_POS, update_reveal, ACCENT_COLOR, )
        self._fading_text(3.5, 0.15, "STE: gradiente atravessa Q(w) como identidade", ACCENT_COLOR, ste_reveal, fontsize=8)
        self._fading_text(1.2, 4.1, "atualizado", ACCENT_COLOR, update_reveal, fontsize=9)
        self._equation_near(*self._W_POS, "w ← w - taxa · ∂L/∂w", update_reveal)

