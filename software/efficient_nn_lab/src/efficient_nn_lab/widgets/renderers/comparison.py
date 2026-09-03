"""Renderer for `comparison` — the ANN x BitNet x SNN table, the one demo
that is all text and therefore sized from the canvas rather than fixed."""

from __future__ import annotations

import textwrap


from efficient_nn_lab.app.theme import (
    ACCENT_COLOR,
    BITNET_COLOR,
    CONVERGE_COLOR,
    NEUTRAL_COLOR,
    SNN_COLOR,
)
from efficient_nn_lab.widgets.renderers._painting import _SKELETON_ALPHA


class ComparisonRendererMixin:
    """Draws the `comparison_pipeline` frame kind."""

    # ================================================================
    # comparison — a persistent, growing 3-column table + outputs panel
    #
    # This is the one demo that is *all text*: five rows x three columns of
    # short phrases, plus an outputs row and two footnotes. Its first
    # version placed every cell at a hard-coded data coordinate with a
    # hard-coded fontsize=8 -- which is what "the SNN and ANN comparison
    # fonts are too small" (FIXME.md) reports. Two separate causes:
    #
    #   1. A fixed point size cannot be right for this widget. The Qt
    #      matplotlib canvas keeps its dpi constant and grows the figure's
    #      *inches* as the window grows, so a fontsize=8 cell that looked
    #      acceptable on a 900x400 canvas is 8pt out of a ~1000pt-tall
    #      figure once the app is on a projector -- i.e. it shrinks,
    #      relatively, exactly when legibility matters most. Measured: the
    #      app hands this widget ~1425x1012 px at a normal window size.
    #   2. Even at fontsize 8 the long cells nearly touched their
    #      neighbours, so "just raise the number" would have collided.
    #
    # So the table sizes *itself*. Every coordinate below is an axes
    # fraction (xlim/ylim are 0..1 for this renderer) and the type size is
    # whichever of the two real constraints binds first:
    #
    #   vertical:   every line of text plus its gaps must fit the box
    #   horizontal: a wrapped cell must fit inside one column
    #
    # Whatever height the type could not claim is then spent on the *gaps*
    # (`spread`), so the table fills its box at any canvas size instead of
    # overflowing small ones and hugging the top edge of large ones.
    # ================================================================
    #: Axes box (figure fractions) for this renderer -- see render().
    _CMP_AX_RECT = (0.02, 0.03, 0.96, 0.94)
    #: Width of the row-name column, as a fraction of the axes box. The
    #: three value columns split what is left, equally.
    _CMP_LABEL_FRAC = 0.20
    _CMP_COL_ORDER = ("ANN", "BitNet", "SNN")
    _CMP_COL_COLORS = {"ANN": NEUTRAL_COLOR, "BitNet": BITNET_COLOR, "SNN": SNN_COLOR}
    #: Characters per line inside a value column. 19 is the smallest
    #: budget that still wraps every cell of comparison/ann_bitnet_snn.py
    #: to at most TWO lines ("quantizada conforme" / "arquitetura" is the
    #: binding case -- at 18 it splits into three, which would collide
    #: with the row below). The row-height budget assumes two.
    _CMP_WRAP = 19
    #: Characters per line for the row names ("Operação principal" ->
    #: two lines). Wide enough to keep "Representação" on one line.
    _CMP_LABEL_WRAP = 14
    #: Average glyph advance as a fraction of the font size, for DejaVu
    #: Sans on these mixed-case strings. Measured from a rendered frame
    #: ("normalmente implícito": 21 chars spanning 123pt at fontsize 11 =
    #: 0.53 em/char); 0.56 leaves a margin.
    _CMP_EM_PER_CHAR = 0.56
    #: Baseline-to-baseline distance as a multiple of the font size.
    _CMP_LINE_H = 1.18
    #: The vertical budget, split into what the font fixes and what may be
    #: stretched. Both are in units of "one line of cell text" and both
    #: must be kept in step with the placement cursor in
    #: _render_comparison_pipeline (the overlap test in
    #: tests/test_widgets_no_clipping.py is what catches a drift).
    #:   text: header 1.35 + 5 rows x 2 + outputs 1 + 2 notes x 0.95
    #:   gaps: top .1, header-rule .35, rule-row .5, 4 x .45 between rows,
    #:         row-rule .55, rule-outputs .45, outputs-note .5,
    #:         note-note .3, bottom .15
    _CMP_TEXT_LINES = 1.35 + 10.0 + 1.0 + 1.9
    _CMP_GAP_LINES = 0.1 + 0.35 + 0.5 + 1.8 + 0.55 + 0.45 + 0.5 + 0.3 + 0.15
    #: Legibility clamps on the cell size. The floor is deliberately below
    #: anything usable: it exists only so an absurdly small embedding still
    #: lays out inside its box instead of spilling out of it. The ceiling
    #: stops a very tall canvas from producing comically large type.
    _CMP_FS_MIN = 5.0
    _CMP_FS_MAX = 26.0
    #: Header/footnote sizes, relative to the computed cell size.
    _CMP_HEADER_RATIO = 1.35
    _CMP_NOTE_RATIO = 0.95
    #: Ceiling on the gap stretch. Past ~3x the rows stop reading as one
    #: table and start reading as unrelated lines.
    _CMP_SPREAD_MAX = 3.0

    def _cmp_geometry(self) -> dict[str, float]:
        """Type sizes and the gap stretch that fit this canvas.

        Single source of truth for the layout: change _CMP_WRAP, add a row
        or retune a gap and the type size follows from it automatically.
        """
        fig_w_in, fig_h_in = self._figure.get_size_inches()
        ax_w_pt = fig_w_in * self._CMP_AX_RECT[2] * 72.0
        ax_h_pt = fig_h_in * self._CMP_AX_RECT[3] * 72.0
        col_frac = (1.0 - self._CMP_LABEL_FRAC) / len(self._CMP_COL_ORDER)

        # the two constraints, each solved for the cell font size
        total_lines = self._CMP_TEXT_LINES + self._CMP_GAP_LINES
        fs_vertical = ax_h_pt / (total_lines * self._CMP_LINE_H)
        fs_horizontal = (ax_w_pt * col_frac * 0.92) / (self._CMP_WRAP * self._CMP_EM_PER_CHAR)
        fs = max(self._CMP_FS_MIN, min(self._CMP_FS_MAX, min(fs_vertical, fs_horizontal)))

        # the footnotes are single ~92-character lines spanning the whole
        # width, so they are width-bound by a different budget than the
        # cells -- sized separately, or they run off both edges.
        fs_note = min(fs * self._CMP_NOTE_RATIO, ax_w_pt / (92 * self._CMP_EM_PER_CHAR))

        line = fs * self._CMP_LINE_H / ax_h_pt  # one line of cell text, axes fraction
        # Height the type could not use (because the *width* limited it)
        # goes into the gaps: text height is fixed by the font, so solve
        # text + spread * gaps = available.
        available_lines = 1.0 / line if line > 0 else total_lines
        spread = (available_lines - self._CMP_TEXT_LINES) / self._CMP_GAP_LINES
        spread = max(1.0, min(self._CMP_SPREAD_MAX, spread))

        return {
            "fs": fs,
            "fs_header": fs * self._CMP_HEADER_RATIO,
            "fs_note": fs_note,
            "line": line,
            "spread": spread,
            "col_frac": col_frac,
        }

    def _render_comparison_pipeline(self, values: dict[str, object]) -> None:
        self._reset_axes(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        g = self._cmp_geometry()
        fs, fs_header, fs_note = g["fs"], g["fs_header"], g["fs_note"]
        line, sp, col_frac = g["line"], g["spread"], g["col_frac"]
        note_line = line * self._CMP_NOTE_RATIO

        cols_x = {
            name: self._CMP_LABEL_FRAC + col_frac * (i + 0.5)
            for i, name in enumerate(self._CMP_COL_ORDER)
        }
        label_x = 0.004

        table = values["table"]
        row_order = list(values["table_rows"])
        reveal_map = {
            "Representação": values["reveal_repr"],
            "Ativação": values["reveal_activation"],
            "Domínio temporal": values["reveal_domain"],
            "Treinamento": values["reveal_training"],
            "Operação principal": values["reveal_operation"],
        }

        # top-down placement cursor, in axes fractions: `text` consumes
        # what the font needs, `gap` consumes stretchable space.
        cursor = 1.0

        def gap(lines: float) -> None:
            nonlocal cursor
            cursor -= lines * line * sp

        def text(height_lines: float) -> float:
            """Consume a text block and return the y its centre sits at."""
            nonlocal cursor
            centre = cursor - height_lines * line / 2.0
            cursor -= height_lines * line
            return centre

        gap(0.1)
        header_y = text(self._CMP_HEADER_RATIO)
        for name, x in cols_x.items():
            self._fading_text(x, header_y, name, self._CMP_COL_COLORS[name], 1.0, fontsize=fs_header, weight="bold")
        # a hairline under the column heads: at this type size the table
        # needs one cue that the top row is a *header*, not just another
        # (differently coloured) row of values.
        gap(0.35)
        self._cmp_rule(cursor, 1.0)
        gap(0.5)

        for index, row_name in enumerate(row_order):
            if index:
                gap(0.45)
            reveal = float(reveal_map[row_name])
            y = text(2.0)
            # the row label stays faintly visible before its row is
            # revealed (alpha floor 0.12) so the table's full height is
            # apparent from the first checkpoint -- the same "skeleton
            # first" rule the node diagrams follow.
            self._fading_text(
                label_x, y, textwrap.fill(row_name, self._CMP_LABEL_WRAP), "black", max(reveal, 0.12),
                fontsize=fs, weight="bold", ha="left",
            )
            for name, cell in zip(self._CMP_COL_ORDER, table[row_name]):
                self._fading_text(
                    cols_x[name], y, textwrap.fill(cell, self._CMP_WRAP),
                    self._CMP_COL_COLORS[name], reveal, fontsize=fs,
                )

        reveal_outputs = float(values["reveal_outputs"])
        gap(0.55)
        self._cmp_rule(cursor, reveal_outputs)
        gap(0.45)
        outputs_y = text(1.0)
        if reveal_outputs > 0.02:
            # "Saída" spelled out: this label used to read just "O", which
            # is not an abbreviation of anything a student can decode.
            self._fading_text(
                label_x, outputs_y, "Saída", "black", reveal_outputs,
                fontsize=fs, weight="bold", ha="left",
            )
            outputs = {
                "ANN": f"y = {values['y_ann']:g}",
                "BitNet": f"y = {values['y_bitnet']:g}",
                "SNN": f"{values['snn_spike_count']} spikes",
            }
            for name, text_value in outputs.items():
                self._fading_text(cols_x[name], outputs_y, text_value, self._CMP_COL_COLORS[name], reveal_outputs, fontsize=fs)

        gap(0.5)
        note1_y = cursor - note_line / 2.0
        cursor -= note_line
        self._fading_text(
            0.5, note1_y,
            "Tipo de gradiente: exato (ANN) / STE (BitNet) / substituto (SNN) — análogos, não idênticos.",
            ACCENT_COLOR, float(values["reveal_gradients"]), fontsize=fs_note,
        )
        gap(0.3)
        note2_y = cursor - note_line / 2.0
        self._fading_text(
            0.5, note2_y,
            "Eficiência não é garantida pela arquitetura: depende de hardware, memória e workload.",
            SNN_COLOR, float(values["reveal_caveat"]), fontsize=fs_note, weight="bold",
        )

    def _cmp_rule(self, y: float, alpha: float) -> None:
        """The table's horizontal hairlines (header underline, outputs divider)."""
        if alpha <= 0.02:
            return
        self._ax.plot(
            [0.0, 1.0], [y, y], color=NEUTRAL_COLOR,
            linewidth=1.0, alpha=_SKELETON_ALPHA * alpha, solid_capstyle="butt",
        )
