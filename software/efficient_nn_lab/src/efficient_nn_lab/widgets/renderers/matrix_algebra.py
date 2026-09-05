"""Renderer for `backprop.matrix` — the same network as a graph and as
real matrices, linked cell-by-edge."""

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


def grid(arr, rv, hl, color, **kw) -> dict:
    """One matrix on the board: its numbers, how much of it is revealed, what glows.

    `rv` is per-cell so a matrix can fill in one entry at a time, and `hl`
    drives the same glow on the graph edge, which is the correspondence this
    demo exists to show.
    """
    return {"t": "grid", "arr": arr, "rv": rv, "hl": hl, "color": color, **kw}


def scalar(value, rv, color, hl=None, **kw) -> dict:
    """A single number, drawn as a 1x1 grid so it lines up with the matrices."""
    return grid(
        np.array([[value]]), np.array([[rv]]),
        np.zeros((1, 1)) if hl is None else np.array([[hl]]), color, dec=4, **kw
    )


class MatrixAlgebraRendererMixin:
    """Draws the `matrix_algebra` frame kind."""

    # ================================================================
    # backprop/demos/matrix_algebra.py — the same network drawn twice at
    # once: a graph on the left, real vectors/matrices on the right. The
    # whole point is that the two are linked cell-by-edge, so a single
    # highlight field (hl_w1[i, j]) glows BOTH the matrix cell and the graph
    # edge x_j -> H_i. Everything is driven by per-cell reveal arrays, which
    # core/math_utils.tween_values interpolates element-wise -- that is what
    # makes a matrix fill in one entry at a time instead of appearing whole.
    #
    # Each node carries its own values INSIDE its box (name / z / y, each
    # with its own alpha). They used to be printed underneath, which put
    # them in the path of the O -> L arrow and forced the boxes to stay
    # small to leave room; in-box means bigger boxes, bigger type, and no
    # arrow striking through a number.
    # ================================================================
    #: Column spacing has to exceed the two nodes' radii plus a visible
    #: stretch of arrow, or the trimmed edge between adjacent columns comes
    #: out zero-length and the connection simply disappears.
    _MA_NODES = {
        "x1": (0.90, 8.15), "x2": (0.90, 5.75),
        "h1": (3.25, 8.15), "h2": (3.25, 5.75),
        "o": (5.60, 6.95),
    }
    _MA_LOSS = (5.60, 4.05)
    _MA_IN_W, _MA_IN_H = 1.05, 0.95     # 2 lines: name + value
    _MA_MID_W, _MA_MID_H = 1.32, 1.35   # 3 lines: name + z + y
    _MA_LOSS_W, _MA_LOSS_H = 1.7, 0.95  # 2 lines: alvo + L
    #: node "radius" used to trim edges to the box outline (boxstyle pad
    #: 0.25 inflates every box beyond its nominal width, so this is
    #: half-width + that pad); without trimming, the semi-transparent boxes
    #: let the arrows show through and cross the numbers inside.
    _MA_RADIUS = {"x1": 0.775, "x2": 0.775, "h1": 0.91, "h2": 0.91, "o": 0.91}

    #: The board is a row of items ([matriz] · [entrada] = [saída]) laid out
    #: by _board_row, not at fixed columns: the matrices change shape between
    #: phases (2x2, 1x2, 2x1, 1x1), and fixed columns tuned for the widest
    #: case leave the operators jammed against the brackets of the narrow
    #: ones. Widths are measured, then the row is centred.
    _MA_ROW1_Y = 7.7    # the multiplication / main operation
    _MA_ACCUM_Y = 5.72  # running partial sum, clear of row 1's captions
    _MA_ROW2_Y = 4.18   # the activation (its own row: it is its own step)
    _MA_BOARD_CX = 11.6
    _MA_BOARD_GAP = 0.24
    _MA_OP_W, _MA_ARROW_W = 0.5, 0.95
    _MA_CELL_W, _MA_CELL_H, _MA_CELL_FS = 1.22, 0.84, 10.0
    _MA_CHAIN_X = (7.70, 9.55, 11.40, 13.25, 15.10)
    _MA_CHAIN_W = 1.2

    def _mx_grid(
        self, cx: float, cy: float, arr, reveal, hl, color: str,
        row_labels: tuple[str, ...] = (), col_labels: tuple[str, ...] = (),
        decimals: int = 2, caption: str = "",
    ) -> None:
        """A bracketed numeric grid whose cells reveal/glow independently.

        ``reveal`` and ``hl`` are same-shape arrays of 0..1: reveal fades a
        cell's number in (0 leaves a faint placeholder dash, so the grid's
        *shape* is legible before any value exists), hl paints the
        highlight chip behind it. Brackets are always drawn -- the empty
        matrix is part of the picture from frame one.
        """
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        reveal = np.atleast_2d(np.asarray(reveal, dtype=float)).reshape(arr.shape)
        hl = np.atleast_2d(np.asarray(hl, dtype=float)).reshape(arr.shape)
        nrows, ncols = arr.shape
        total_w, total_h = ncols * self._MA_CELL_W, nrows * self._MA_CELL_H
        x0, x1 = cx - total_w / 2, cx + total_w / 2
        y0, y1 = cy - total_h / 2, cy + total_h / 2

        for bx, tick in ((x0 - 0.11, 0.2), (x1 + 0.11, -0.2)):
            self._ax.plot([bx, bx], [y0, y1], color=color, linewidth=2.2, alpha=0.9, solid_capstyle="butt")
            self._ax.plot([bx, bx + tick], [y1, y1], color=color, linewidth=2.2, alpha=0.9, solid_capstyle="butt")
            self._ax.plot([bx, bx + tick], [y0, y0], color=color, linewidth=2.2, alpha=0.9, solid_capstyle="butt")

        for i in range(nrows):
            for j in range(ncols):
                ccx = x0 + (j + 0.5) * self._MA_CELL_W
                ccy = y1 - (i + 0.5) * self._MA_CELL_H
                if hl[i, j] > 0.02:
                    self._ax.add_patch(FancyBboxPatch(
                        (ccx - self._MA_CELL_W / 2 + 0.06, ccy - self._MA_CELL_H / 2 + 0.05),
                        self._MA_CELL_W - 0.12, self._MA_CELL_H - 0.1,
                        facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.55 * float(hl[i, j]),
                        boxstyle="round,pad=0.04", zorder=1,
                    ))
                rv = float(reveal[i, j])
                if rv > 0.02:
                    self._ax.text(
                        ccx, ccy, f"{arr[i, j]:+.{decimals}f}", ha="center", va="center",
                        fontsize=self._MA_CELL_FS, family="monospace", color="black",
                        alpha=rv, zorder=3,
                    )
                if rv < 0.98:
                    self._ax.text(
                        ccx, ccy, "·", ha="center", va="center", fontsize=11,
                        color=NEUTRAL_COLOR, alpha=_SKELETON_ALPHA * (1.0 - rv), zorder=2,
                    )

        for j, label in enumerate(col_labels):
            self._ax.text(
                x0 + (j + 0.5) * self._MA_CELL_W, y1 + 0.22, label, ha="center", va="bottom",
                fontsize=9, color=NEUTRAL_COLOR, style="italic",
            )
        for i, label in enumerate(row_labels):
            self._ax.text(
                x0 - 0.42, y1 - (i + 0.5) * self._MA_CELL_H, label, ha="right", va="center",
                fontsize=9, color=NEUTRAL_COLOR, style="italic",
            )
        if caption:
            self._ax.text(
                cx, y0 - 0.32, caption, ha="center", va="top", fontsize=9.5,
                color=color, style="italic",
            )

    def _node_box(
        self, pos: tuple[float, float], w: float, h: float, color: str,
        lines: tuple[tuple[str, float], ...], box_alpha: float, glow: float = 0.0,
        fontsize: float = 9.5,
    ) -> None:
        """A graph node whose text lines each fade in on their own.

        ``lines`` is ``((text, alpha), ...)`` top to bottom. Separate alphas
        are what let the animation show "z is computed but y is not yet" --
        a single box label could only appear all at once.
        """
        x, y = pos
        if glow > 0.02:
            self._ax.add_patch(FancyBboxPatch(
                (x - w / 2 - 0.13, y - h / 2 - 0.13), w + 0.26, h + 0.26,
                facecolor=ACCENT_COLOR, edgecolor="none", alpha=0.5 * glow,
                boxstyle="round,pad=0.25",
            ))
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, facecolor=color, alpha=_FILL_ALPHA * box_alpha,
            edgecolor=color, **_BOX_STYLE,
        )
        self._ax.add_patch(patch)
        n = len(lines)
        step = h / max(1, n)
        for k, (text, alpha) in enumerate(lines):
            if alpha <= 0.02:
                continue
            ly = y + h / 2 - step * (k + 0.5)
            self._ax.text(
                x, ly, text, ha="center", va="center", fontsize=fontsize if k == 0 else fontsize - 1.2,
                color="black", alpha=alpha, zorder=4,
                fontweight="bold" if k == 0 else "normal",
            )

    def _ma_edge(self, src: str, dest: str, glow: float) -> None:
        """One graph edge, trimmed to both node outlines."""
        n = self._MA_NODES
        p0, p1 = np.array(n[src], dtype=float), np.array(n[dest], dtype=float)
        direction = p1 - p0
        length = float(np.hypot(*direction))
        unit = direction / length
        a = p0 + unit * self._MA_RADIUS[src]
        b = p1 - unit * self._MA_RADIUS[dest]
        self._skeleton_arrow(tuple(a), tuple(b))
        self._flow_arrow(tuple(a), tuple(b), glow, ACCENT_COLOR)

    def _board_row(self, items: list[dict], y: float) -> None:
        """Lay a row of grids/operators out left-to-right, centred on the board.

        ``items`` are dicts tagged by ``t``: ``"grid"`` (keys arr/rv/hl/color
        plus optional rows/cols/dec/cap), ``"op"`` (a glyph between two
        grids) or ``"arrow"`` (a flow arrow with a glyph above it, used for
        the σ / ½()² step). Each item's width is measured from its actual
        shape first, so operators always land in the middle of the gap
        whatever the matrices' dimensions are.
        """
        widths = []
        for item in items:
            if item["t"] == "op":
                widths.append(self._MA_OP_W)
            elif item["t"] == "arrow":
                widths.append(self._MA_ARROW_W)
            elif item["t"] == "grid":
                ncols = np.atleast_2d(np.asarray(item["arr"], dtype=float)).shape[1]
                widths.append(ncols * self._MA_CELL_W + 0.22)
            else:
                raise ValueError(f"_board_row: unknown item type {item['t']!r}")
        total = sum(widths) + self._MA_BOARD_GAP * (len(items) - 1)
        x = self._MA_BOARD_CX - total / 2.0
        for item, width in zip(items, widths):
            cx = x + width / 2.0
            if item["t"] == "grid":
                self._mx_grid(
                    cx, y, item["arr"], item["rv"], item["hl"], item["color"],
                    row_labels=item.get("rows", ()), col_labels=item.get("cols", ()),
                    decimals=item.get("dec", 2), caption=item.get("cap", ""),
                )
            elif item["t"] == "op":
                self._fading_text(
                    cx, y, item["glyph"], NEUTRAL_COLOR, item.get("alpha", 1.0),
                    fontsize=item.get("size", 24 if item["glyph"] == "·" else 19), weight="bold",
                )
            else:
                reveal = item.get("alpha", 1.0)
                self._skeleton_arrow((x, y), (x + width, y))
                self._flow_arrow((x, y), (x + width, y), reveal, item["color"])
                self._fading_text(
                    cx, y + 0.52, item["glyph"], item["color"], reveal,
                    fontsize=item.get("size", 15), weight="bold",
                )
            x += width + self._MA_BOARD_GAP

    def _render_matrix_algebra(self, values: dict[str, object]) -> None:
        """One frame of the matrix-algebra demo: graph, algebra, running sum."""
        self._reset_axes(xlim=(-0.1, 16.1), ylim=(0.4, 10.0))
        self._render_graph_panel(values)
        self._render_algebra_board(values)
        self._render_running_sum(values)

    def _render_graph_panel(self, values: dict[str, object]) -> None:
        """The network as a graph, on the left.

        `hl_w1`/`hl_w2` are the SAME arrays that glow the matrix cells on
        the right. That is the whole point of this demo: one weight, two
        pictures, lit together.
        """
        x = values["x"]
        z1 = values["z1"]
        y1 = values["y1"]
        z2 = float(values["z2"])
        y2 = float(values["y2"])
        target = float(values["target"])
        loss = float(values["loss"])
        rv_graph = float(values["rv_graph"])
        rv_x = values["rv_x"]
        rv_z1 = values["rv_z1"]
        rv_y1 = values["rv_y1"]
        rv_z2 = float(values["rv_z2"])
        rv_y2 = float(values["rv_y2"])
        rv_target = float(values["rv_target"])
        rv_loss = float(values["rv_loss"])
        hl_w1 = values["hl_w1"]
        hl_w2 = values["hl_w2"]
        hl_x = values["hl_x"]
        hl_y1 = values["hl_y1"]
        hl_out = float(values["hl_out"])
        n = self._MA_NODES
        # -- the graph, left. hl_w1/hl_w2 are the SAME arrays that glow the
        # matrix cells, which is the mapping this demo is about.
        for i, dest in enumerate(("h1", "h2")):
            for j, src in enumerate(("x1", "x2")):
                self._ma_edge(src, dest, float(hl_w1[i, j]))
        for j, src in enumerate(("h1", "h2")):
            self._ma_edge(src, "o", float(hl_w2[0, j]))
        o_bottom = (n["o"][0], n["o"][1] - self._MA_MID_H / 2 - 0.25)
        loss_top = (self._MA_LOSS[0], self._MA_LOSS[1] + self._MA_LOSS_H / 2 + 0.25)
        self._skeleton_arrow(o_bottom, loss_top)
        self._flow_arrow(o_bottom, loss_top, rv_loss, SNN_COLOR)

        self._node_box(
            n["x1"], self._MA_IN_W, self._MA_IN_H, NEUTRAL_COLOR,
            (("x1", rv_graph), (f"{float(x[0]):+.2f}", float(rv_x[0]))),
            rv_graph, glow=float(hl_x[0]),
        )
        self._node_box(
            n["x2"], self._MA_IN_W, self._MA_IN_H, NEUTRAL_COLOR,
            (("x2", rv_graph), (f"{float(x[1]):+.2f}", float(rv_x[1]))),
            rv_graph, glow=float(hl_x[1]),
        )
        for key, label, idx in (("h1", "H1", 0), ("h2", "H2", 1)):
            self._node_box(
                n[key], self._MA_MID_W, self._MA_MID_H, BITNET_COLOR,
                (
                    (label, rv_graph),
                    (f"z={float(z1[idx]):+.3f}", float(rv_z1[idx])),
                    (f"y={float(y1[idx]):.3f}", float(rv_y1[idx])),
                ),
                rv_graph, glow=float(hl_y1[idx]),
            )
        self._node_box(
            n["o"], self._MA_MID_W, self._MA_MID_H, CONVERGE_COLOR,
            (("O", rv_graph), (f"z={z2:+.3f}", rv_z2), (f"y={y2:.3f}", rv_y2)),
            rv_graph, glow=hl_out,
        )
        self._node_box(
            self._MA_LOSS, self._MA_LOSS_W, self._MA_LOSS_H, SNN_COLOR,
            ((f"alvo = {target:g}", rv_target), (f"L = {loss:.4f}", rv_loss)),
            max(rv_target, rv_loss),
        )
        self._fading_text(3.25, 9.6, "a rede como grafo", NEUTRAL_COLOR, 1.0, fontsize=10.5, weight="bold")


    def _render_algebra_board(self, values: dict[str, object]) -> None:
        """The algebra, on the right, one operation per frame.

        `focus` says which operation this frame is showing, and each mode
        draws a different board. They were one 143-line if/elif chain;
        making them separate methods is what lets each say, in its own
        docstring, which derivative it is."
        """
        focus = str(values["focus"])
        # -- the algebra board, right. Row 1 is the multiplication; row 2 is
        # the activation, which gets its own row because it is its own step.
        self._fading_text(
            self._MA_BOARD_CX, 9.6, str(values["board_title"]), BITNET_COLOR, 1.0,
            fontsize=11.5, weight="bold",
        )

        if focus in ("chain",):
            self._board_chain(values)
            return

        if focus in ("l1", "l2"):
            self._board_layers(values)
            return

        if focus in ("loss",):
            self._board_loss(values)
            return

        if focus in ("gz2",):
            self._board_gz2(values)
            return

        if focus in ("gw2",):
            self._board_gw2(values)
            return

        if focus in ("w2t",):
            self._board_w2t(values)
            return

        if focus in ("gz1",):
            self._board_gz1(values)
            return

        if focus in ("gw1",):
            self._board_gw1(values)
            return

        # No silent fallback (CLAUDE.md): a focus nobody draws would render an
        # empty board, and an empty board looks like a step that simply has
        # nothing to show rather than a step that was never written.
        raise ValueError(f"matrix_algebra: unknown focus {focus!r}")


    def _board_chain(self, values: dict[str, object]) -> None:
        """The five chain-rule factors, walked one at a time."""
        row1 = self._MA_ROW1_Y
        row2 = self._MA_ROW2_Y
        gw1 = values["gw1"]
        names, vals = values["chain_names"], values["chain_values"]
        rv_chain = values["rv_chain"]
        for k, cx in enumerate(self._MA_CHAIN_X):
            self._node_box(
                (cx, row1), self._MA_CHAIN_W, 1.15, BITNET_COLOR,
                ((names[k], float(rv_chain[k])), (f"{vals[k]:+.4f}", float(rv_chain[k]))),
                float(rv_chain[k]), fontsize=9.5,
            )
            if k:
                self._fading_text(
                    (cx + self._MA_CHAIN_X[k - 1]) / 2, row1, "·", NEUTRAL_COLOR,
                    float(rv_chain[k]), fontsize=26, weight="bold",
                )
        self._node_box(
            (self._MA_BOARD_CX, self._MA_ACCUM_Y), 4.8, 0.8, ACCENT_COLOR,
            ((f"produto = {float(values['chain_product']):+.5f}", 1.0),),
            float(values["rv_chain_product"]), fontsize=11,
        )
        self._node_box(
            (self._MA_BOARD_CX, row2), 5.6, 0.8, CONVERGE_COLOR,
            ((f"= grad_W1[H1,x1] = {float(np.asarray(gw1)[0, 0]):+.5f}  ✓", 1.0),),
            float(values["rv_check"]), fontsize=11,
        )


    def _board_layers(self, values: dict[str, object]) -> None:
        """A layer's matrix product and its activation, side by side."""
        row1 = self._MA_ROW1_Y
        row2 = self._MA_ROW2_Y
        focus = str(values["focus"])
        x = values["x"]
        w1 = values["w1"]
        w2 = values["w2"]
        z1 = values["z1"]
        y1 = values["y1"]
        z2 = float(values["z2"])
        y2 = float(values["y2"])
        rv_x = values["rv_x"]
        rv_w1 = values["rv_w1"]
        rv_w2 = values["rv_w2"]
        rv_z1 = values["rv_z1"]
        rv_y1 = values["rv_y1"]
        rv_z2 = float(values["rv_z2"])
        rv_y2 = float(values["rv_y2"])
        hl_w1 = values["hl_w1"]
        hl_w2 = values["hl_w2"]
        hl_x = values["hl_x"]
        hl_y1 = values["hl_y1"]
        first = focus == "l1"
        out_vec = z1 if first else np.array([z2])
        act_vec = y1 if first else np.array([y2])
        rv_out = rv_z1 if first else np.array([rv_z2])
        rv_act = rv_y1 if first else np.array([rv_y2])
        self._board_row([
            grid(
                w1 if first else w2, rv_w1 if first else rv_w2, hl_w1 if first else hl_w2,
                BITNET_COLOR, rows=("H1", "H2") if first else ("O",),
                cols=("x1", "x2") if first else ("H1", "H2"),
                cap="W1 (2×2)" if first else "W2 (1×2)",
            ),
            {"t": "op", "glyph": "·"},
            grid(
                (x if first else y1).reshape(-1, 1),
                np.asarray(rv_x if first else rv_y1).reshape(-1, 1),
                np.asarray(hl_x if first else hl_y1).reshape(-1, 1),
                NEUTRAL_COLOR, cap="x" if first else "y (camada 1)",
            ),
            {"t": "op", "glyph": "="},
            grid(
                out_vec.reshape(-1, 1), np.asarray(rv_out).reshape(-1, 1),
                np.zeros((len(out_vec), 1)), CONVERGE_COLOR, dec=3, cap="z",
            ),
        ], row1)
        # the activation, on its own row -- one step of the walkthrough
        act_reveal = float(np.max(np.asarray(rv_act)))
        self._board_row([
            grid(
                out_vec.reshape(-1, 1), np.asarray(rv_out).reshape(-1, 1),
                np.zeros((len(out_vec), 1)), CONVERGE_COLOR, dec=3, cap="z",
            ),
            {"t": "arrow", "glyph": "σ", "color": ACCENT_COLOR, "alpha": act_reveal},
            grid(
                act_vec.reshape(-1, 1), np.asarray(rv_act).reshape(-1, 1),
                np.zeros((len(act_vec), 1)), CONVERGE_COLOR, dec=3, cap="y = σ(z)",
            ),
        ], row2)


    def _board_loss(self, values: dict[str, object]) -> None:
        """The error, its square, and the halving that cancels the 2."""
        row1 = self._MA_ROW1_Y
        row2 = self._MA_ROW2_Y
        zero = np.zeros((1, 1))
        y2 = float(values["y2"])
        target = float(values["target"])
        diff = float(values["diff"])
        loss = float(values["loss"])
        rv_y2 = float(values["rv_y2"])
        rv_target = float(values["rv_target"])
        rv_diff = float(values["rv_diff"])
        rv_loss = float(values["rv_loss"])
        hl_out = float(values["hl_out"])
        self._board_row([
            scalar(y2, rv_y2, CONVERGE_COLOR, hl=hl_out, cap="y_O (rede)"),
            {"t": "op", "glyph": "−"},
            grid(np.array([[target]]), np.array([[rv_target]]), zero, NEUTRAL_COLOR, dec=2, cap="alvo"),
            {"t": "op", "glyph": "="},
            scalar(diff, rv_diff, SNN_COLOR, cap="diferença"),
        ], row1)
        self._board_row([
            scalar(diff, rv_diff, SNN_COLOR, cap="diferença"),
            {"t": "arrow", "glyph": "½( )²", "color": SNN_COLOR, "alpha": rv_loss, "size": 12},
            scalar(loss, rv_loss, SNN_COLOR, cap="L"),
        ], row2)


    def _board_gz2(self, values: dict[str, object]) -> None:
        """dL/dz at the output: the incoming gradient times sigma'."""
        row1 = self._MA_ROW1_Y
        sp2 = float(values["sp2"])
        gy2 = float(values["gy2"])
        gz2 = float(values["gz2"])
        rv_sp2 = float(values["rv_sp2"])
        hl_out = float(values["hl_out"])
        self._board_row([
            scalar(gy2, float(values["rv_gy2"]), SNN_COLOR, cap="∂L/∂y_O"),
            {"t": "op", "glyph": "·"},
            scalar(sp2, rv_sp2, ACCENT_COLOR, hl=hl_out, cap="σ'(z_O)"),
            {"t": "op", "glyph": "="},
            scalar(gz2, float(values["rv_gz2"]), SNN_COLOR, cap="∂L/∂z_O"),
        ], row1)


    def _board_gw2(self, values: dict[str, object]) -> None:
        """dL/dW2: the output delta against the hidden activations."""
        row1 = self._MA_ROW1_Y
        y1 = values["y1"]
        gz2 = float(values["gz2"])
        gw2 = values["gw2"]
        hl_y1 = values["hl_y1"]
        self._board_row([
            scalar(gz2, 1.0, SNN_COLOR, cap="∂L/∂z_O"),
            {"t": "op", "glyph": "⊗"},
            grid(y1.reshape(1, -1), np.ones((1, 2)), np.asarray(hl_y1).reshape(1, -1), CONVERGE_COLOR, dec=3, cap="y (linha)"),
            {"t": "op", "glyph": "="},
            grid(gw2, values["rv_gw2"], np.zeros((1, 2)), SNN_COLOR, rows=("O",), cols=("H1", "H2"), dec=4, cap="grad_W2 (1×2)"),
        ], row1)


    def _board_w2t(self, values: dict[str, object]) -> None:
        """W2 transposed -- how the gradient travels back a layer."""
        row1 = self._MA_ROW1_Y
        w2 = values["w2"]
        gz2 = float(values["gz2"])
        gy1 = values["gy1"]
        rv_w2 = values["rv_w2"]
        hl_w2 = values["hl_w2"]
        self._board_row([
            grid(w2.T, np.asarray(rv_w2).T, np.asarray(hl_w2).T, BITNET_COLOR, rows=("H1", "H2"), cols=("O",), cap="W2ᵀ (2×1)"),
            {"t": "op", "glyph": "·"},
            scalar(gz2, 1.0, SNN_COLOR, cap="∂L/∂z_O"),
            {"t": "op", "glyph": "="},
            grid(gy1.reshape(-1, 1), np.asarray(values["rv_gy1"]).reshape(-1, 1), np.zeros((2, 1)), SNN_COLOR, dec=4, cap="∂L/∂y"),
        ], row1)


    def _board_gz1(self, values: dict[str, object]) -> None:
        """dL/dz in the hidden layer, once sigma' is applied."""
        row1 = self._MA_ROW1_Y
        sp1 = values["sp1"]
        gy1 = values["gy1"]
        gz1 = values["gz1"]
        rv_sp1 = values["rv_sp1"]
        hl_y1 = values["hl_y1"]
        self._board_row([
            grid(gy1.reshape(-1, 1), np.ones((2, 1)), np.zeros((2, 1)), SNN_COLOR, rows=("H1", "H2"), dec=4, cap="∂L/∂y"),
            {"t": "op", "glyph": "⊙"},
            grid(
                np.asarray(sp1).reshape(-1, 1), np.asarray(rv_sp1).reshape(-1, 1),
                np.asarray(hl_y1).reshape(-1, 1), ACCENT_COLOR, dec=4, cap="σ'(z)",
            ),
            {"t": "op", "glyph": "="},
            grid(gz1.reshape(-1, 1), np.asarray(values["rv_gz1"]).reshape(-1, 1), np.zeros((2, 1)), SNN_COLOR, dec=4, cap="∂L/∂z"),
        ], row1)


    def _board_gw1(self, values: dict[str, object]) -> None:
        """dL/dW1: the hidden delta against the inputs."""
        row1 = self._MA_ROW1_Y
        x = values["x"]
        gz1 = values["gz1"]
        gw1 = values["gw1"]
        hl_x = values["hl_x"]
        hl_y1 = values["hl_y1"]
        self._board_row([
            grid(
                gz1.reshape(-1, 1), np.ones((2, 1)), np.asarray(hl_y1).reshape(-1, 1),
                SNN_COLOR, rows=("H1", "H2"), dec=4, cap="∂L/∂z (coluna)",
            ),
            {"t": "op", "glyph": "⊗"},
            grid(x.reshape(1, -1), np.ones((1, 2)), np.asarray(hl_x).reshape(1, -1), NEUTRAL_COLOR, cap="x (linha)"),
            {"t": "op", "glyph": "="},
            grid(gw1, values["rv_gw1"], np.zeros((2, 2)), SNN_COLOR, rows=("H1", "H2"), cols=("x1", "x2"), dec=4, cap="grad_W1 (2×2)"),
        ], row1)


    def _render_running_sum(self, values: dict[str, object]) -> None:
        """The partial sum, interpolated as a real number.

        It counts up while the transition plays instead of jumping to the
        answer, which is the difference between showing a sum being built
        and showing a sum already built.
        """
        focus = str(values["focus"])
        # -- the running partial sum: a real interpolated number, so it
        # visibly counts up/down while the transition plays instead of
        # jumping to the answer.
        rv_accum = float(values["rv_accum"])
        if rv_accum > 0.02 and focus != "chain":
            self._node_box(
                (self._MA_BOARD_CX, self._MA_ACCUM_Y), 4.6, 0.8, ACCENT_COLOR,
                ((f"soma parcial = {float(values['accum']):+.4f}", 1.0),), rv_accum, fontsize=11,
            )

        work = str(values["work_text"])
        if work:
            self._ax.text(
                0.0, 2.95, work, ha="left", va="top", fontsize=9.2, color="black",
                family="monospace", linespacing=1.55,
            )


