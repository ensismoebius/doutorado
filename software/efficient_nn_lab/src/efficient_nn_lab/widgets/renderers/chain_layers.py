"""Renderer for `backprop.chain` — the chain-rule ladder: one block of the
network per column, that block's local derivatives underneath it."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch

from efficient_nn_lab.app.theme import (
    ACCENT_COLOR,
    BITNET_COLOR,
    CONVERGE_COLOR,
    NEUTRAL_COLOR,
    SNN_COLOR,
)
from efficient_nn_lab.widgets.renderers._painting import _BOX_STYLE, _FILL_ALPHA, _SKELETON_ALPHA


class ChainLayersRendererMixin:
    """Draws the `chain_layers` frame kind."""

    # ================================================================
    # backprop/demos/chain_rule_layers.py — the correspondence between the
    # layers and the chain rule, drawn as a ladder:
    #
    #   row 1  the forward pass as SIX blocks, not two neurons: a layer is
    #          a linear op (z = w·in + b) *and* an activation (a = σ(z)),
    #          and the chain rule treats each as its own link;
    #   row 2  that block's parameters, hanging under it;
    #   row 3+ that block's LOCAL DERIVATIVES, one card per input, in the
    #          same column as the block. This vertical alignment is the
    #          whole point of the demo -- one block, one factor -- so the
    #          cards are positioned from the same _CL_NODE_X list as the
    #          blocks rather than from their own coordinates, and a dotted
    #          connector is drawn down each column;
    #   row 4  the accumulated product δ, one per layer, at that layer's z;
    #   row 5  the four parameter gradients (w1, b1, w2, b2 -- none left
    #          implicit: the bias cards read "= 1", which is exactly why
    #          ∂L/∂b = δ);
    #   row 6  the five factors of ∂L/∂w1 in a strip, in multiplication
    #          order, with the running product -- the concatenation seen
    #          whole after being seen block by block.
    #
    # Card row 0 is the backward "highway" (the factor that continues left);
    # rows 1 and 2 are the branches to that block's parameters. The
    # right-to-left arrows are drawn only along row 0, and each one fills
    # when the value at its head appears.
    # ================================================================
    _CL_AX_RECT = (0.025, 0.02, 0.95, 0.95)
    _CL_XLIM = (-0.35, 16.6)
    #: FancyBboxPatch's ``boxstyle=round,pad=0.25`` inflates every box by
    #: 0.25 DATA UNITS on each side, so a chip declared 0.95 high actually
    #: occupies 1.45. Every pitch below budgets for that.
    #:
    #: The vertical span is the real constraint of this drawing: the canvas
    #: gives ~685pt of height for twelve rows, and the type has to fit
    #: inside boxes that are measured in units. The row heights below are
    #: what is left after paying for legible type (see _CL_FS_*), not the
    #: other way round -- the first version had it backwards and the text
    #: came out too small to read.
    _CL_YLIM = (0.0, 15.0)
    #: One x per forward block, in the demo's NODE_* order
    #: (x, z1, a1, z2, a2, L). Blocks, parameter chips, derivative cards, δ
    #: chips and gradient chips all read their x from here, so a column
    #: cannot drift out of alignment with the block it belongs to.
    _CL_NODE_X = (1.10, 3.85, 6.60, 9.35, 12.10, 14.85)
    _CL_COL_PITCH = 2.75
    _CL_NODE_Y = 12.55
    _CL_NODE_W, _CL_NODE_H = 1.95, 1.05
    _CL_CAPTION_Y = 13.92
    _CL_BAND_Y0, _CL_BAND_Y1 = 10.10, 14.95
    _CL_BAND_LABEL_Y = 14.68
    _CL_PARAM_Y = 10.95
    _CL_PARAM_W, _CL_PARAM_H = 0.95, 0.62
    _CL_PARAM_DX = 0.78
    _CL_DIVIDER_Y = 9.72
    _CL_DIVIDER_LABEL_Y = 9.93
    _CL_CARD_ROW_Y = (8.45, 6.95, 5.45)
    _CL_CARD_W, _CL_CARD_H = 1.95, 0.95
    _CL_DELTA_Y = 4.05
    # δ and the gradient chips stay INSIDE their block's column
    # (_CL_COL_PITCH): a chip wider than the column it belongs to
    # spills over the neighbouring shading and undoes the grouping the
    # column is there to make.
    _CL_DELTA_W, _CL_DELTA_H = 2.60, 0.75
    _CL_GRAD_Y = 2.78
    # The two gradient chips of a block are the one row that has to run
    # slightly wider than its column: keeping them inside would force the
    # value down to ~10pt. Harmless -- at this height the neighbouring
    # columns (the activations) are empty, so nothing is overprinted.
    _CL_GRAD_W, _CL_GRAD_H = 1.35, 0.72
    _CL_GRAD_DX = 0.98
    _CL_STRIP_Y = 1.55
    #: shifted right of the row's caption ("cadeia de ∂L/∂w1:"), whose
    #: drawn width grew with the type: the first chip used to sit on it.
    _CL_STRIP_X = (2.85, 4.90, 6.95, 9.00, 11.05)
    _CL_STRIP_W, _CL_STRIP_H = 1.35, 0.70
    _CL_PRODUCT_X, _CL_PRODUCT_W = 14.35, 3.70
    _CL_WORK_Y = 0.93
    #: The backward columns are shaded from the divider down to the bottom
    #: of the gradient row, in the colour of the block they belong to. A
    #: dotted connector alone was too faint to answer "which card belongs
    #: to which block?" from the back of a room -- a filled column with a
    #: dashed outline answers it without being read.
    _CL_COLUMN_TOP = 9.55
    _CL_COLUMN_BOTTOM = 1.98
    _CL_COLUMN_ALPHA = 0.10

    #: Type sizes, at the reference canvas (see _cl_type_scale). Sized
    #: against the drawn box, which is 0.5 data units WIDER and taller
    #: than its nominal w/h (boxstyle pad), and verified by rendering:
    #: the binding element is the card label "∂a2/∂z2 = σ'(z2)", the
    #: longest string that has to fit one column. Labels are
    #: deliberately a size below their values: the number is what gets read
    #: from a distance, the symbol is what gets read once.
    _CL_FS_BAND = 14.0
    _CL_FS_CAPTION = 12.5
    _CL_FS_NODE_LABEL, _CL_FS_NODE_VALUE = 15.0, 19.0
    _CL_FS_PARAM_LABEL, _CL_FS_PARAM_VALUE = 11.5, 14.5
    _CL_FS_CARD_LABEL, _CL_FS_CARD_VALUE = 13.0, 17.0
    _CL_FS_DELTA_LABEL, _CL_FS_DELTA_VALUE = 14.0, 17.5
    _CL_FS_GRAD_LABEL, _CL_FS_GRAD_VALUE = 12.5, 15.0
    _CL_FS_STRIP_LABEL, _CL_FS_STRIP_VALUE = 11.5, 14.5
    _CL_FS_DIVIDER = 12.5
    _CL_FS_NOTE = 11.5
    _CL_FS_WORK = 11.5

    #: (first block, last block, label) for the shaded bands that group the
    #: blocks into layers. This is the "correspondence with the layers" made
    #: visible: the band shows that z1+a1 are ONE layer in two blocks.
    _CL_BANDS = ((0, 0, "entrada"), (1, 2, "camada 1 (oculta)"), (3, 4, "camada 2 (saída)"), (5, 5, "perda"))
    #: Caption printed above each block: what the block computes.
    _CL_CAPTIONS = (
        "entrada (dado)",
        "operação linear\nz1 = w1·x + b1",
        "ativação\na1 = σ(z1)",
        "operação linear\nz2 = w2·a1 + b2",
        "ativação\na2 = σ(z2)",
        "perda\nL = ½(a2 − alvo)²",
    )
    #: One name/colour per forward block, in _CL_NODE_X order.
    _CL_NODE_NAMES = ("x", "z1", "a1", "z2", "a2", "L")
    _CL_NODE_COLORS = (NEUTRAL_COLOR, BITNET_COLOR, CONVERGE_COLOR, BITNET_COLOR, CONVERGE_COLOR, SNN_COLOR)
    #: Parameter chips: (block, slot, label, value key, reveal key). Slot -1
    #: is left of the block's centre, +1 right, 0 centred.
    _CL_PARAMS = (
        (1, -1, "w1", "w1", "rv_w1"),
        (1, +1, "b1", "b1", "rv_b1"),
        (3, -1, "w2", "w2", "rv_w2"),
        (3, +1, "b2", "b2", "rv_b2"),
        (5, 0, "alvo", "target", "rv_target"),
    )
    #: Local-derivative cards: (block, row, label, value key, reveal key,
    #: decimals). Row 0 is the backward highway; rows 1-2 the parameter
    #: branches of that block.
    _CL_CARDS = (
        (5, 0, "∂L/∂a2", "dL_da2", "rv_dL_da2", 4),
        (4, 0, "∂a2/∂z2 = σ'(z2)", "sp2", "rv_sp2", 4),
        (3, 0, "∂z2/∂a1 = w2", "dz2_da1", "rv_dz2_da1", 2),
        (3, 1, "∂z2/∂w2 = a1", "dz2_dw2", "rv_dz2_dw2", 4),
        (3, 2, "∂z2/∂b2", "dz2_db2", "rv_dz2_db2", 2),
        (2, 0, "∂a1/∂z1 = σ'(z1)", "sp1", "rv_sp1", 4),
        (1, 1, "∂z1/∂w1 = x", "dz1_dw1", "rv_dz1_dw1", 2),
        (1, 2, "∂z1/∂b1", "dz1_db1", "rv_dz1_db1", 2),
    )
    #: Right-to-left arrows along the highway: (from block, to block, the
    #: reveal field of the value that lands at the head).
    _CL_HIGHWAY = ((5, 4, "rv_sp2"), (4, 3, "rv_delta2"), (3, 2, "rv_dL_da1"), (2, 1, "rv_delta1"))
    #: δ chips: (block, label, value key, reveal key).
    _CL_DELTAS = ((3, "δ2 = ∂L/∂z2", "delta2", "rv_delta2"), (1, "δ1 = ∂L/∂z1", "delta1", "rv_delta1"))
    #: Gradient chips: (block, slot, label, value key, reveal key).
    _CL_GRADS = (
        (3, -1, "∂L/∂w2", "g_w2", "rv_g_w2"),
        (3, +1, "∂L/∂b2", "g_b2", "rv_g_b2"),
        (1, -1, "∂L/∂w1", "g_w1", "rv_g_w1"),
        (1, +1, "∂L/∂b1", "g_b1", "rv_g_b1"),
    )

    #: Canvas height (in points, inside _CL_AX_RECT) this drawing's point
    #: sizes were tuned against: the ~1012px-tall canvas the app actually
    #: hands this widget at a normal window size.
    _CL_REFERENCE_H_PT = 10.12 * 0.95 * 72.0

    def _cl_type_scale(self) -> float:
        """Factor applied to every point size in this renderer.

        Every box here is positioned and sized in DATA units, so it shrinks
        with the widget -- but a hard-coded point size does not. At a small
        canvas the labels then spill out of their boxes and over each other
        (the same failure the comparison table had, from the other
        direction). Scaling the type by the canvas keeps text and box in
        proportion at any size; the clamps stop it from becoming
        unreadably small or absurdly large.
        """
        h_pt = self._figure.get_size_inches()[1] * self._CL_AX_RECT[3] * 72.0
        return max(0.45, min(1.5, h_pt / self._CL_REFERENCE_H_PT))

    def _cl_band_x(self, first: int, last: int) -> tuple[float, float]:
        half = self._CL_NODE_W / 2 + 0.32
        return self._CL_NODE_X[first] - half, self._CL_NODE_X[last] + half

    def _cl_chip(
        self, x: float, y: float, w: float, h: float, color: str,
        label: str, value: str, reveal: float, glow: float = 0.0,
        label_fs: float = 11.0, value_fs: float = 14.0,
    ) -> None:
        """A named quantity and its number, as one chip that fades in.

        Used for parameters, derivative cards, δ and gradients alike --
        drawing them all here is what keeps the rows visually consistent.
        The two lines get independent sizes on purpose: the *number* is
        what has to be readable from the back of the room, the symbol only
        has to be readable once, up close.
        """
        self._skeleton_box(x, y, w=w, h=h)
        if reveal <= 0.02:
            return
        if glow > 0.02:
            self._ax.add_patch(FancyBboxPatch(
                (x - w / 2 - 0.13, y - h / 2 - 0.13), w + 0.26, h + 0.26,
                facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.55 * glow,
                boxstyle="round,pad=0.25",
            ))
        self._ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, facecolor=color, alpha=_FILL_ALPHA * reveal,
            edgecolor=color, **_BOX_STYLE,
        ))
        self._ax.text(
            x, y + h * 0.23, label, ha="center", va="center", fontsize=label_fs,
            color="black", alpha=reveal, fontweight="bold", zorder=4,
        )
        self._ax.text(
            x, y - h * 0.26, value, ha="center", va="center", fontsize=value_fs,
            color="black", alpha=reveal, zorder=4,
        )

    def _cl_draw_bands(self, rv_graph: float, scale: float) -> None:
        """The layer bands: the "correspondence with the layers".

        z1 and a1 sit inside ONE band, which is what says "these two blocks
        are the same layer" without a word of text.
        """
        for first, last, label in self._CL_BANDS:
            band_x0, band_x1 = self._cl_band_x(first, last)
            self._ax.add_patch(FancyBboxPatch(
                (band_x0, self._CL_BAND_Y0), band_x1 - band_x0, self._CL_BAND_Y1 - self._CL_BAND_Y0,
                facecolor=NEUTRAL_COLOR, edgecolor=NEUTRAL_COLOR, linewidth=1.0,
                alpha=0.10 * max(rv_graph, 0.3), boxstyle="round,pad=0.02", zorder=0,
            ))
            self._fading_text(
                (band_x0 + band_x1) / 2, self._CL_BAND_LABEL_Y, label, NEUTRAL_COLOR,
                max(rv_graph, 0.3), fontsize=self._CL_FS_BAND * scale, weight="bold",
            )

    def _cl_draw_forward_row(
        self, values: dict[str, object], scale: float, rv_graph: float, hl_nodes: np.ndarray
    ) -> None:
        """Forward row: blocks, captions, and the arrows between them."""
        node_x = self._CL_NODE_X
        node_reveal = [float(values[k]) for k in ("rv_x", "rv_z1", "rv_a1", "rv_z2", "rv_a2", "rv_loss")]
        node_value = [
            f"{float(values['x']):+.2f}", f"{float(values['z1']):+.4f}", f"{float(values['a1']):.4f}",
            f"{float(values['z2']):+.4f}", f"{float(values['a2']):.4f}", f"{float(values['loss']):.4f}",
        ]
        half = self._CL_NODE_W / 2 + 0.25  # + boxstyle pad, so arrows do not cross the numbers
        for i in range(len(node_x) - 1):
            a = (node_x[i] + half, self._CL_NODE_Y)
            b = (node_x[i + 1] - half, self._CL_NODE_Y)
            self._skeleton_arrow(a, b)
            self._flow_arrow(a, b, node_reveal[i + 1], NEUTRAL_COLOR)
        for i, x in enumerate(node_x):
            self._fading_text(
                x, self._CL_CAPTION_Y, self._CL_CAPTIONS[i], NEUTRAL_COLOR,
                max(rv_graph, 0.25), fontsize=self._CL_FS_CAPTION * scale,
            )
            self._skeleton_box(x, self._CL_NODE_Y, w=self._CL_NODE_W, h=self._CL_NODE_H)
            if rv_graph > 0.02:
                self._cl_chip(
                    x, self._CL_NODE_Y, self._CL_NODE_W, self._CL_NODE_H, self._CL_NODE_COLORS[i],
                    self._CL_NODE_NAMES[i], node_value[i] if node_reveal[i] > 0.02 else "",
                    rv_graph, glow=float(hl_nodes[i]),
                    label_fs=self._CL_FS_NODE_LABEL * scale, value_fs=self._CL_FS_NODE_VALUE * scale,
                )

    def _cl_draw_params(self, values: dict[str, object], scale: float, hl_params: np.ndarray) -> None:
        """Parameters, hanging under the block that uses them.

        Index into hl_params is the position in _CL_PARAMS, which is the
        demo's PARAM_* order -- the two lists are one contract.
        """
        node_x = self._CL_NODE_X
        for slot_index, (block, slot, label, value_key, reveal_key) in enumerate(self._CL_PARAMS):
            self._cl_chip(
                node_x[block] + slot * self._CL_PARAM_DX, self._CL_PARAM_Y,
                self._CL_PARAM_W, self._CL_PARAM_H, NEUTRAL_COLOR,
                label, f"{float(values[value_key]):+.2f}", float(values[reveal_key]),
                glow=float(hl_params[slot_index]),
                label_fs=self._CL_FS_PARAM_LABEL * scale, value_fs=self._CL_FS_PARAM_VALUE * scale,
            )

    def _cl_draw_divider_and_columns(self, rv_graph: float, rv_summary: float, scale: float) -> None:
        """The divider line, its caption, and the shaded backward columns.

        Each backward column is SHADED in its block's colour, from the
        divider down past the gradients, and joined to the block above it
        by a line in the same colour. The first version used a faint
        dotted line alone and it did not answer "which card belongs to
        which block?" at a glance -- a filled column answers it without
        being read, which is the whole claim of this demo.
        """
        node_x = self._CL_NODE_X
        self._ax.plot(
            [self._CL_XLIM[0] + 0.2, self._CL_XLIM[1] - 0.2], [self._CL_DIVIDER_Y] * 2,
            color=NEUTRAL_COLOR, linewidth=1.2, alpha=_SKELETON_ALPHA, linestyle="--",
        )
        # The divider caption and the closing "general rule" banner want the
        # same full-width strip, and only one of them is ever the point of
        # the step -- so the caption fades out exactly as the banner fades
        # in, instead of the two overprinting each other.
        self._fading_text(
            self._CL_XLIM[0] + 0.25, self._CL_DIVIDER_LABEL_Y,
            "backward: a derivada local de cada bloco, na coluna do bloco  ←",
            NEUTRAL_COLOR, max(rv_graph, 0.3) * (1.0 - rv_summary),
            fontsize=self._CL_FS_DIVIDER * scale, weight="bold", ha="left",
        )

        columns = {block for block, *_ in self._CL_CARDS}
        half = self._CL_COL_PITCH / 2 - 0.06
        for block in sorted(columns):
            x = node_x[block]
            colour = self._CL_NODE_COLORS[block]
            alpha = self._CL_COLUMN_ALPHA * max(rv_graph, 0.35)
            self._ax.add_patch(FancyBboxPatch(
                (x - half, self._CL_COLUMN_BOTTOM), 2 * half,
                self._CL_COLUMN_TOP - self._CL_COLUMN_BOTTOM,
                facecolor=colour, edgecolor=colour, linewidth=1.2, linestyle="--",
                alpha=alpha, boxstyle="round,pad=0.02", zorder=0,
            ))
            # the neck joining the block to its own column
            self._ax.plot(
                [x, x],
                [self._CL_NODE_Y - self._CL_NODE_H / 2 - 0.3, self._CL_COLUMN_TOP],
                color=colour, linewidth=2.0, alpha=0.45 * max(rv_graph, 0.35),
                linestyle=(0, (2, 2)), zorder=0,
            )

    def _cl_draw_highway_and_cards(
        self, values: dict[str, object], scale: float, rv_graph: float, hl_cards: np.ndarray
    ) -> None:
        """The right-to-left highway arrows and the local-derivative cards."""
        node_x = self._CL_NODE_X
        for block, to_block, reveal_key in self._CL_HIGHWAY:
            fill = float(values[reveal_key])
            a = (node_x[block] - self._CL_CARD_W / 2 - 0.05, self._CL_CARD_ROW_Y[0])
            b = (node_x[to_block] + self._CL_CARD_W / 2 + 0.05, self._CL_CARD_ROW_Y[0])
            self._skeleton_arrow(a, b)
            self._flow_arrow(a, b, fill, SNN_COLOR)

        for block, row, label, value_key, reveal_key, decimals in self._CL_CARDS:
            value = float(values[value_key])
            self._cl_chip(
                node_x[block], self._CL_CARD_ROW_Y[row], self._CL_CARD_W, self._CL_CARD_H,
                ACCENT_COLOR, label, f"{value:+.{decimals}f}", float(values[reveal_key]),
                glow=float(hl_cards[block, row]),
                label_fs=self._CL_FS_CARD_LABEL * scale, value_fs=self._CL_FS_CARD_VALUE * scale,
            )
        # The highway's last arrow points into this empty slot on purpose:
        # the derivative ∂z1/∂x exists, but x is data, so the backward pass
        # has nothing left to do there. Saying it on screen turns the empty
        # slot from a confusion ("did a card fail to appear?") into the
        # lesson, and keeps "no parameter skipped" honest -- x is not one.
        self._fading_text(
            node_x[1], self._CL_CARD_ROW_Y[0],
            "∂z1/∂x = w1 existe,\nmas x é dado —\naqui o backward para",
            NEUTRAL_COLOR, 0.6 * max(rv_graph, 0.2), fontsize=self._CL_FS_NOTE * scale,
        )

    def _cl_draw_deltas_and_grads(self, values: dict[str, object], scale: float) -> None:
        """The δ chips (backward accumulator) and the parameter-gradient chips."""
        node_x = self._CL_NODE_X
        for block, label, value_key, reveal_key in self._CL_DELTAS:
            reveal = float(values[reveal_key])
            self._cl_chip(
                node_x[block], self._CL_DELTA_Y, self._CL_DELTA_W, self._CL_DELTA_H,
                SNN_COLOR, label, f"{float(values[value_key]):+.5f}", reveal,
                label_fs=self._CL_FS_DELTA_LABEL * scale, value_fs=self._CL_FS_DELTA_VALUE * scale,
            )
        for block, slot, label, value_key, reveal_key in self._CL_GRADS:
            self._cl_chip(
                node_x[block] + slot * self._CL_GRAD_DX, self._CL_GRAD_Y,
                self._CL_GRAD_W, self._CL_GRAD_H, SNN_COLOR,
                label, f"{float(values[value_key]):+.6f}", float(values[reveal_key]),
                label_fs=self._CL_FS_GRAD_LABEL * scale, value_fs=self._CL_FS_GRAD_VALUE * scale,
            )

    def _cl_draw_chain_strip(self, values: dict[str, object], scale: float, rv_graph: float) -> None:
        """The same five factors, now in multiplication order, plus the product check."""
        names = values["chain_names"]
        factors = values["chain_values"]
        rv_chain = np.asarray(values["rv_chain"], dtype=float)
        self._fading_text(
            self._CL_XLIM[0] + 0.25, self._CL_STRIP_Y, "cadeia de ∂L/∂w1:", NEUTRAL_COLOR,
            max(rv_graph, 0.3), fontsize=self._CL_FS_STRIP_LABEL * scale, weight="bold", ha="left",
        )
        for k, x in enumerate(self._CL_STRIP_X):
            self._cl_chip(
                x, self._CL_STRIP_Y, self._CL_STRIP_W, self._CL_STRIP_H, ACCENT_COLOR,
                str(names[k]), f"{float(factors[k]):+.4f}", float(rv_chain[k]),
                label_fs=self._CL_FS_STRIP_LABEL * scale, value_fs=self._CL_FS_STRIP_VALUE * scale,
            )
            if k:
                self._fading_text(
                    (x + self._CL_STRIP_X[k - 1]) / 2, self._CL_STRIP_Y, "·", NEUTRAL_COLOR,
                    float(rv_chain[k]), fontsize=24 * scale, weight="bold",
                )
        rv_product = float(values["rv_chain_product"])
        self._fading_text(
            self._CL_STRIP_X[-1] + self._CL_STRIP_W / 2 + 0.35, self._CL_STRIP_Y, "=",
            NEUTRAL_COLOR, rv_product, fontsize=20 * scale, weight="bold",
        )
        rv_check = float(values["rv_check"])
        self._skeleton_box(self._CL_PRODUCT_X, self._CL_STRIP_Y, w=self._CL_PRODUCT_W, h=self._CL_STRIP_H)
        if rv_product > 0.02:
            self._node_box(
                (self._CL_PRODUCT_X, self._CL_STRIP_Y), self._CL_PRODUCT_W, self._CL_STRIP_H,
                CONVERGE_COLOR,
                (
                    (f"produto = {float(values['chain_product']):+.6f}", rv_product),
                    (f"= δ1 · x = ∂L/∂w1 = {float(values['g_w1']):+.6f}  ✓", rv_check),
                ),
                rv_product, fontsize=self._CL_FS_STRIP_VALUE * scale,
            )

    def _cl_draw_summary_and_work(
        self, values: dict[str, object], scale: float, rv_summary: float
    ) -> None:
        """The general rule, revealed last, and the numeric work-shown text.

        The general rule goes on the divider row, in the width the
        left-hand caption leaves free -- the only band of the drawing that
        is empty at every step, so revealing it collides with nothing.
        """
        if rv_summary > 0.02:
            self._fading_text(
                (self._CL_XLIM[0] + self._CL_XLIM[1]) / 2, self._CL_DIVIDER_LABEL_Y,
                "por camada, para trás:  δ → δ · w · σ'(z)    |    por parâmetro:  ∂L/∂w = δ · entrada,   ∂L/∂b = δ · 1",
                ACCENT_COLOR, rv_summary, fontsize=self._CL_FS_DIVIDER * scale, weight="bold",
            )

        work = str(values["work_text"])
        if work:
            self._ax.text(
                self._CL_XLIM[0] + 0.25, self._CL_WORK_Y, work, ha="left", va="top",
                fontsize=self._CL_FS_WORK * scale, color="black", family="monospace", linespacing=1.3,
            )

    def _render_chain_layers(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=self._CL_XLIM, ylim=self._CL_YLIM)
        scale = self._cl_type_scale()  # every fontsize below is scaled by this
        rv_graph = float(values["rv_graph"])
        rv_summary = float(values["rv_summary"])
        hl_nodes = np.asarray(values["hl_nodes"], dtype=float)
        hl_params = np.asarray(values["hl_params"], dtype=float)
        hl_cards = np.asarray(values["hl_cards"], dtype=float)

        self._cl_draw_bands(rv_graph, scale)
        self._cl_draw_forward_row(values, scale, rv_graph, hl_nodes)
        self._cl_draw_params(values, scale, hl_params)
        self._cl_draw_divider_and_columns(rv_graph, rv_summary, scale)
        self._cl_draw_highway_and_cards(values, scale, rv_graph, hl_cards)
        self._cl_draw_deltas_and_grads(values, scale)
        self._cl_draw_chain_strip(values, scale, rv_graph)
        self._cl_draw_summary_and_work(values, scale, rv_summary)

