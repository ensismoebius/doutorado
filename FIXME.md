# FIXME plan — efficient_nn_lab software issues

Scope: `software/efficient_nn_lab/`. Four issues reported in one batch. Each
section below is investigation findings + a concrete step-by-step fix plan.
Nothing in this file has been implemented yet — this is the plan only.

---

## 1. "BitNet -> Backward -> STE" doesn't explain step-by-step / numerically

**File:** `src/efficient_nn_lab/bitnet/demos/backward.py`
**Widgets involved:** `widgets/neuron_view.py::_render_ste_pipeline` (L581-611),
`widgets/weight_view.py::_render_staircase` (L74-115) and
`_render_quant_derivative` (L118-163).

### Root cause (confirmed by reading the code)

This demo has three scenes. Two of them (`staircase`, `quant_derivative`)
already show *some* numbers (`τ = 0.50`, `dQ/dw = 0.00`), but neither is tied
to a concrete example weight — the "current" derivative value plotted in
`_render_quant_derivative` (L129: `curve[len(curve) // 2]`) is just the
midpoint of the sampled curve, not a value the explanation text ever
references.

The third scene — the forward/backward path diagram, `_render_ste_pipeline`
— is the actual root problem. Compare it to every other pipeline diagram in
the app:

- `_render_backprop_pipeline` boxes show `"z = {values['z']:g}"`,
  `"y = σ(z) = {values['y']:.3f}"`, etc. — real numbers.
- `_render_forward_pipeline` boxes show real `x`, `w`, quantized `w`,
  products, `y`, `loss`.
- `_render_ste_pipeline` boxes show only **static generic labels**:
  `"peso real"`, `"quantização"`, `"peso ternário"`, `"operação"`, `"loss"`,
  `"gradiente"`, `"STE"` (L577, L579). No field in `path_frame()`
  (`backward.py` L68-74) carries a numeric value at all — only three reveal
  floats (`fwd`, `bwd`, `joined`).

So the demo titled "Backward -> STE" — the one place a student needs to see
*an actual weight* go through quantization, get a *real* loss gradient, and
watch the STE substitute a *specific* number (1) for the *real* local
derivative (0) — never displays a single number in its main diagram. The
explanation text (`backward.py` L96-155) is entirely qualitative too: no
`f"{w:g}"`-style interpolation anywhere in the file, unlike `forward.py` and
`traditional_gd.py` which are full of it.

### Plan

1. Give `BackwardSTEDemo` a concrete example weight, e.g. `self.w = 0.65`
   (a parameter, slider-adjustable, mirroring `ForwardLossDemo`'s pattern),
   plus a `self.upstream_grad` (dL/dy from "outside", so the STE math has
   something concrete to propagate) — or reuse `bitnet.linear`/`bitnet.ste`
   helpers the way `forward.py`/`traditional_gd.py` do, so the numbers are
   computed by the real quantization/STE code, not hand-typed.
2. Rewrite `path_frame()` (L68-74) to carry the real values through
   (`w`, `w_quant`, `upstream_grad`, `dQ/dw_real=0`, `dQ/dw_ste=1`,
   `dL/dw_real≈0`, `dL/dw_ste=upstream_grad`), same pattern as
   `_base_values()` in `forward.py` (L34-59).
3. Rewrite `_render_ste_pipeline` (`neuron_view.py` L581-611) so each box's
   label includes its real value, e.g. `"peso real\nw = 0.65"`,
   `"peso ternário\nQ(w) = +1"`, `"gradiente\ndL/dQ(w) = -0.30"`,
   `"STE\ndQ(w)/dw := 1"` — same box-with-value convention as every other
   pipeline diagram in the file.
4. Rewrite every `explanation=` string in `backward.py` (L96-155) to
   interpolate the real numbers, same style as `forward.py`'s `reason()`
   helper (L96-101) and `traditional_gd.py`'s per-step f-strings — e.g.
   "w = 0.65 está acima de τ = 0.50, então Q(w) = +1" instead of the current
   purely conceptual text.
5. Tie `_render_quant_derivative`'s "current point" (`weight_view.py` L129)
   to the same example `w` instead of the curve's midpoint, so the three
   scenes reference the *same* concrete weight throughout — one worked
   example, not three unrelated abstractions.
6. Verify with real numbers before wiring into the widget (repo convention,
   see CLAUDE.md `/didactic-explanation`): compute the worked example in a
   throwaway Python snippet first, confirm `Q(w)`, `dL/dw` values by hand,
   *then* embed them in the f-strings.
7. Update/extend `tests/test_bitnet.py` to assert the new numeric fields
   are present and correct on `BackwardSTEDemo`'s checkpoints (mirrors the
   existing hand-worked-numbers tests for `forward.py`/`traditional_gd.py`
   in `tests/test_bitnet.py` / `tests/test_backprop.py`).
8. Re-render all checkpoints (offscreen) and eyeball a few screenshots to
   confirm the boxes are legible with the added value text (font size /
   box width may need adjusting, `_box()`'s `w=2.0` might be tight for a
   two-line label).

---

## 2. Animation members must never be drawn outside / clipped

### Investigation performed this session

Wrote an automated check (not yet committed) that renders every checkpoint
frame of every demo and compares each figure's `get_tightbbox()` against its
canvas pixel bounds — i.e. "did any artist (box, arrow, text, image) end up
outside what the widget will actually display."

Ran it two ways:
- Standalone widgets at a generous fixed size (900x550): **0 overflow**
  across all 12 demos, ~1000+ checkpoint frames.
- Through the real `MainWindow` (`resize(1180, 720)`, demos selected via
  `_select_demo()` exactly like a real click), which produces canvas sizes
  ranging **226px to 419px tall** depending on demo (explanation-label
  height + controls eat into the stack's available space differently per
  demo): **still 0 overflow**, all 12 demos, all checkpoints.

So there is **no reproducible clipping under normal window sizes** right
now. Two real gaps found, though:

- **`MainWindow` has no `setMinimumSize()` / `setMinimumHeight()`**
  (`app/main_window.py`, checked around L126-260 — grep confirms no
  `setMinimumSize` anywhere in the file). Nothing stops the user from
  resizing the window down to a tiny size, at which point fixed-point-size
  fonts (all `fontsize=` calls in `neuron_view.py`/`weight_view.py`/
  `signal_view.py` are absolute points, not scaled to canvas size) will
  eventually overflow their boxes/axes for real. Not reproduced yet, but
  structurally guaranteed to happen below some threshold.
- Several box/text placements sit **very close to their axes' declared
  xlim/ylim** with little margin, e.g. in `_render_backprop_pipeline`
  (`neuron_view.py`): `_BP_Y` box (`w=1.9` centered at `x=5.6`, L405) spans
  `x ∈ [4.65, 6.55]` against `xlim=(-0.2, 6.6)` — 0.05 units of margin;
  `_BP_GRAD_Z`'s equation text (`dy=-0.5` from `y=0.15`, L470) lands at
  `y=-0.35` against `ylim=(-0.3, 6.7)` — already past the nominal axis
  edge, currently absorbed by the axes' own figure-position padding but
  fragile to any future layout change.

### Plan

1. Commit the tight-bbox overflow check as a real test,
   `tests/test_widgets_no_clipping.py`: for every demo class, for every
   checkpoint frame, render into the appropriate widget at 2-3 canvas sizes
   (a generous one, a cramped one matching the smallest size seen through
   `MainWindow`, and a "stress" size smaller than anything seen today) and
   assert `get_tightbbox()` stays within canvas bounds. This turns "no
   clipping" from a one-off finding into a permanent regression guard —
   every future demo/widget change gets checked automatically.
2. Add `MainWindow.setMinimumSize(...)` (`app/main_window.py`, in
   `__init__` near L129's `self.resize(1180, 720)`) sized so the canvas
   never drops below the smallest size the stress test in step 1 passes at.
   Pick the floor empirically: shrink in the stress test until content
   genuinely starts overflowing, set the minimum comfortably above that.
3. Widen the tight spots found above defensively, even though not
   currently overflowing: bump `_BP_Y`'s box width down slightly or widen
   `xlim`/`ylim` in `_render_backprop_pipeline`'s `_reset_axes` call
   (`neuron_view.py` L413) by ~0.3-0.5 units on the tight sides.
4. Re-run the full test suite + the new clipping test after each of the
   three demos/widgets flagged tight, confirm still 0 overflow at both the
   normal and the new stress size.

---

## 3. "AGAIN, I can't see the Patrick image!!!!"

### What was verified this session (and in the prior session per summary)

- `resources/images/patrick.jpg` exists, loads correctly
  (`load_grayscale_image`, `snn/encoding.py` L64-79): shape `(108, 192)`,
  `std ≈ 0.29`, full `[0.015, 0.992]` range — genuinely not a blank/flat
  image.
- `PoissonImageCodingDemo._build_frames()` (`poisson_image_coding.py`
  L59-85) wires the loaded image into every frame correctly.
- Routing is correct: `"poisson_image_coding"` is in `_SIGNAL_KINDS`
  (`app/main_window.py` L52-55), so `_choose_view` sends it to
  `signal_view`, which dispatches to `_render_poisson_image`
  (`signal_view.py` L134-161).
- Rendered the real `MainWindow` (offscreen, but through the actual Qt
  widget stack, real `_select_demo`/`_refresh_frame` calls, a real
  `QPixmap` grab) on `snn.poisson_image`'s first checkpoint: **Patrick is
  clearly visible** in the grabbed screenshot.
- `matplotlib.image.imread` on this machine reads the JPEG fine (Pillow
  12.3.0 present; `matplotlib>=3.7`'s own `install_requires` pulls in
  Pillow transitively, confirmed via `pip show matplotlib`).
- `main_window.py` has **no `try`/`except` anywhere** — if image loading
  ever raised (missing Pillow, corrupt file, bad path), selecting this demo
  would crash with a visible traceback, not silently show a blank panel.
  That's a useful diagnostic: if the app is still usable and just shows
  "no image," a hard load failure is *not* the likely cause.

**Bottom line: this cannot be reproduced from the code or in this sandbox.**
Two rounds of investigation (headless tests, integration tests, live-widget
screenshot, and now a full MainWindow walk-through) all show the image
rendering correctly. Something about the user's actual runtime environment
or usage path differs from everything tested so far.

### Plan

1. **Need from the user** (can't proceed without this): run
   `./run.sh --demo snn.poisson_image` directly and report:
   - Any terminal output/traceback (stdout+stderr), especially anything
     mentioning `PIL`, `Pillow`, `patrick.jpg`, or `imread`.
   - A screenshot of exactly what the panel shows (blank white? grey?
     black? distorted? genuinely empty except for the "Spikes sorteados"
     panel below it? wrong demo entirely?).
   - Output of `.venv/bin/pip show pillow matplotlib` in their actual venv
     (was it created before matplotlib started requiring Pillow? is it a
     `--system-site-packages` venv picking up a broken system install?).
   - OS/desktop environment and whether they're on a fresh `.venv` or one
     that predates recent `pip install -e .` runs.
2. **Defensive fix regardless of root cause**: wrap the image load in
   `poisson_image_coding.py::_build_frames` (L60) in a `try/except`, and on
   failure return frames whose `explanation` clearly states the image
   failed to load and why (path, underlying exception message) instead of
   letting `DemoModule.initialize()` (`core/demo.py` L159-166) raise
   uncaught. Turns a currently-ambiguous failure mode into a loud, readable
   one on screen — matches CLAUDE.md's `/didactic-explanation` rule
   ("the failure mode named as loud or silent").
3. Add a regression test asserting the loaded image has real variance
   (`image.std() > 0.05` or similar), not just correct shape/range —
   catches a future regression where the image loads "successfully" but as
   a flat/degenerate array (e.g. wrong channel math, corrupted asset).
4. Once the user's diagnostic info comes back, re-open this section with
   an actual root cause instead of the current "cannot reproduce."

---

## 4. References text has poor contrast

**File:** `app/main_window.py` L212-213 (`self.references_view =
QTextEdit(_REFERENCES_TEXT)`), styled by `app/theme.py`.

### Root cause (confirmed by reading the stylesheet)

`theme.py`'s `STYLESHEET` sets `QWidget { color: {TEXT_COLOR} }` (L48-52,
applies everywhere, including `QTextEdit`) and `QMainWindow { background:
{BACKGROUND} }` (L45-47, applies **only** to the main window itself, not to
descendant widgets). Every other text surface in the app is a `QLabel`,
which paints no background of its own and simply shows the white
`QMainWindow` canvas through it — so it never needed an explicit background
rule.

`QTextEdit` is different: it's a "Base"-colored, self-painting widget by
Qt's own design (like a text editor), and there is **no stylesheet rule for
`QTextEdit` anywhere in `theme.py`**. With no explicit rule, its background
falls back to the OS/Qt-theme palette's "Base" role — which is whatever the
user's desktop/Qt style provides (light in most default setups, but dark
under many Linux dark themes/GTK integrations). Meanwhile its *text* color
is still force-set to `TEXT_COLOR` (`#111318`, near-black) by the generic
`QWidget` rule.

If the user's system palette gives `QTextEdit` a dark "Base" background,
the result is near-black text on a near-black background — exactly "almost
illegible from lack of contrast." This is the **only** `QTextEdit` in the
whole app (confirmed: `grep -n QTextEdit` finds exactly one instantiation),
which is why it's the one place this shows up — every other widget's text
color was WCAG-contrast-verified against the app's own `theme.py` colors in
an earlier session, but that verification pass never covered `QTextEdit`
because it wasn't given app colors to verify against in the first place.

### Plan

1. Add an explicit `QTextEdit` rule to `theme.py`'s `STYLESHEET` (near the
   `QListWidget, QTreeWidget` rule, L53-56, same pattern):
   ```
   QTextEdit {{
       background: {BACKGROUND};
       color: {TEXT_COLOR};
       border: 1px solid #B6C0D6;
       padding: 8px;
   }}
   ```
   This makes the references panel's contrast independent of the user's OS
   theme, matching how every other widget in the app already gets its
   colors from `theme.py` rather than the platform palette.
2. Re-run the WCAG relative-luminance contrast check (same Python
   methodology used earlier this session for the LaTeX deck and the rest
   of the app) on `TEXT_COLOR` (`#111318`) against `BACKGROUND` (`#FFFFFF`)
   — expect comfortably >4.5:1 (it's already used as body text everywhere
   else), but verify rather than assume.
3. Visually confirm via an offscreen `MainWindow` grab (same technique used
   in this session for the Explanation-label fixed-height fix and the
   Patrick-image check): select "Referências" in the tree, grab the
   window, eyeball the text.
4. Check `_REFERENCES_TEXT` (`app/main_window.py`, defined near L58 per the
   `_SIGNAL_KINDS`/`_WEIGHT_KINDS` block) for any inline HTML/rich-text
   color spans that might override the new stylesheet rule locally — none
   expected (it looks like it's assigned as a plain string), but confirm.

---

## 5. Other software improvements found this session

Not reported by the user — found while investigating items 1-4. Listed in
descending order of how much they'd actually help a student/audience
member, not file order.

### 5.1 `surrogate_gradient.py` has the exact same "no numbers" gap as item 1

**File:** `src/efficient_nn_lab/snn/demos/surrogate_gradient.py`

Checked because it's explicitly documented (module docstring, L4-6) as "the
STE story, mirrored" — and it mirrors the bug too. `self.k` (the slope
parameter, L35) never appears in any `explanation=` string; there's no
concrete `(v, gradient)` pair ever stated as a number (e.g. "at v=0.3, the
surrogate gradient = 0.45"); `v_th` is never given a numeric value either.
Zero `f"...{var}..."` numeric interpolation in the whole file (confirmed:
`grep -c '{.*:g}\|{.*\.[0-9]f}'` returns 0), same as `backward.py` before
the item-1 fix. Since item 1's plan establishes the "concrete worked
example, wired through every scene" pattern for the BitNet/STE demo, apply
the identical treatment here for consistency: pick a concrete `v` near
`v_th`, thread `k`, `v`, the real derivative (≈0), and the surrogate
derivative (a real, computed number) through `_render_...`'s two panels
and the explanation text.

### 5.2 Comparison demo's per-row narration is filler text

**File:** `src/efficient_nn_lab/comparison/ann_bitnet_snn.py` L78-82

Every one of the first five checkpoints (one per table row) uses the exact
same templated, content-free sentence:
`f"{row_name}: veja como ANN, BitNet e SNN se comparam nesta linha."`
("...see how they compare in this row") — it names the row but never says
*what* the comparison actually shows. The later checkpoints (outputs,
gradients, caveat, L86-116) are properly specific. Plan: write one real
sentence per row referencing the actual table cell values already sitting
in `table` (L43-49) — e.g. for "Representação":
`"ANN usa ponto flutuante contínuo; BitNet reduz cada peso a {-1,0,+1}; SNN "
"nem guarda um valor contínuo — representa por spikes ao longo do tempo."`

### 5.3 Zero test coverage above the widget-render layer

**Files:** `tests/` (no file references `MainWindow`)

`test_widgets_render.py` confirms every frame renders without raising, but
nothing exercises `MainWindow` itself: view routing when a tree item is
clicked, `_refresh_frame`'s label/equation/detail text updates, the fixed
explanation-label height added this session, lecture-mode/professor-mode
toggling, or the keyboard shortcuts (`_build_shortcuts`, L253-258 area).
This is the layer where this session's own fixes landed (fixed-height
labels, `--demo` deep-linking) with no regression test protecting any of
it. Plan: add `tests/test_main_window.py` using the same
`qapp`/offscreen-platform fixture pattern already used elsewhere, covering
at minimum: selecting a demo routes to the right stack widget, stepping
updates `frame_label`/`explanation_label` text, `explanation_label`'s
height stays constant across frames of different text length (regression
test for this session's fix), and each keyboard shortcut calls the
expected slot.

### 5.4 Keyboard shortcuts are invisible to the user

**File:** `src/efficient_nn_lab/app/main_window.py` `_build_shortcuts`
(L253-258)

Space/←/→/R/Esc are wired up but never surfaced anywhere in the UI — no
tooltip on the corresponding buttons, no visible hint, no help menu.
`grep -n "setToolTip"` across the whole `src/` tree returns nothing. A
first-time user (or an audience member watching over the speaker's
shoulder) has no way to discover these exist. Plan: add
`setToolTip("Space")`-style hints to `ControlsWidget`'s Play/Pause/
Anterior/Próximo/Reset buttons (`widgets/controls.py`) naming their
shortcut, matching how the button text already names the action.

### 5.5 Two labels bypass the app's central theme/contrast system

**File:** `src/efficient_nn_lab/app/main_window.py` L229-230 (`equation_label`,
inline `"font-family: monospace; color: #444;"`) and L234-235
(`detail_label`, inline `"font-family: monospace; font-size: 9pt; color:
#555;"`)

Both colors were checked this session and are contrast-safe on white
(`#444444` → 9.7:1, `#555555` → 7.5:1 against `#FFFFFF`, both well past
4.5:1) — **not a bug**, but these two are the only text colors in the
whole app set via inline `setStyleSheet()` calls instead of
`theme.py`'s centralized `STYLESHEET`, so they're invisible to any future
palette change (e.g. a dark-mode theme) and weren't part of the WCAG audit
done on the rest of the app earlier this session — that audit only found
them because they happened to already pass. Plan: move both into
`theme.py` as `QLabel#Equation` / `QLabel#Detail` rules (same pattern as
the existing `QLabel#Explanation` rule, theme.py L78-80), referencing
named constants instead of ad-hoc hex literals.

### 5.6 "Professor mode" detail panel just dumps `repr(dict)`

**File:** `src/efficient_nn_lab/app/main_window.py` L391

```python
self.detail_label.setText("estado completo: " + repr(frame.values))
```

For demos whose frames carry numpy arrays (e.g. `traditional_gd.py`'s
convergence chart, which stores whole `iterations`/`w`/`y`/`loss`/`z_trail`
arrays per frame, `traditional_gd.py` L267-282) this prints an
unformatted, unreadable wall of `array([...])` text — the opposite of what
"professor mode" (aimed at someone who wants to inspect internals) should
give them. Plan: format `frame.values` field-by-field instead of one
`repr()` call — one line per key, arrays summarized (shape + first/last
few values) rather than fully dumped, matching the didactic, numbers-not-
noise style used everywhere else in the app.

---

## 6. Presentation (LaTeX deck) improvements found this session

**Dir:** `documentation/08-lectures/fronteiras-bitnets-redes-pulso/`

### 6.1 Default beamer navigation symbols were never disabled

**File:** `preamble.tex`

`grep -n "navigation symbols"` finds nothing — `\setbeamertemplate{navigation
symbols}{}` is never called, so beamer's default tiny navigation icon
cluster (the outdated arrow/dot strip) renders in the bottom-right corner
of all 59 pages. Virtually every modern beamer deck disables this; it adds
visual clutter and isn't how anyone actually navigates a PDF during a talk.
Plan: add `\setbeamertemplate{navigation symbols}{}` to `preamble.tex` near
the other `\setbeamertemplate`/`\setbeamercolor` calls (around L178-188),
recompile, confirm visually the icons are gone from a sampled page.

### 6.2 No slide/page counter anywhere in the deck

**File:** `preamble.tex`

No `footline` template is set (`grep -n "footline"` → nothing), so neither
speaker nor audience can see "34/59" on screen. For a 59-page talk with 11
live-software cutovers, this matters twice over: for the speaker's own
pacing, and for audience Q&A ("can we go back to the slide about STE" is
much easier to answer/ask with a visible number). Plan: add a minimal
footline showing `\insertframenumber{} / \inserttotalframenumber`, styled
to stay legible against every one of the five per-section background
tints already in use (verify contrast the same WCAG way used for
`chromeText` earlier this session — a fixed dark/light footline color needs
to clear 4.5:1 against all five, not just white).

### 6.3 One bibliography entry is defined but never cited

**File:** `referencias.bib`

`dan_goodman_2022_7044500` exists in the `.bib` file but no `\cite{...}` in
any `slides/*.tex` references it (checked via key-set comparison against
every `\cite`/`\citep`/etc. call across all 24 slide files — all 23 *other*
entries resolve correctly both ways, only this one is one-directional).
Not a compile error (unused `.bib` entries are silently allowed), just a
dead reference. Plan: either cite it somewhere relevant (Goodman's
`Brian2` framework fits naturally into `snnHardware.tex` or
`snnAplicacoes.tex` if it's SNN-simulator-related) or remove it from
`referencias.bib` if it was leftover from an earlier draft — check what
the entry actually is before deciding which.

### 6.4 No `\hypersetup` — link appearance unverified

**File:** `preamble.tex`

`grep -n "hypersetup\|colorlinks\|urlcolor\|linkcolor"` finds nothing.
`hyperref`'s un-configured default sometimes draws a visible colored
border box around clickable elements in some PDF viewers (though many,
including beamer's own link handling, suppress this by default — **not
confirmed broken**, just never explicitly configured, unlike everything
else color-related in this deck which was deliberately set this session).
Plan: render a page containing one of the `\href{run:...}` "Abrir no
software" links and one `\cite{}` cross-reference at high zoom in at least
two PDF viewers (the one used for the actual talk, plus one other) and
confirm no stray border appears; if one does, add
`\hypersetup{hidelinks}` (or `colorlinks=true` with explicit link colors
matching the rest of the deck's palette) to `preamble.tex`.

### 6.5 PDF metadata Subject/Keywords are empty

**File:** `apresentacao.tex` (`\title`/`\author`/`\date`, L3-11) plus
whatever hyperref config controls `\hypersetup{pdfsubject=...}`
(currently none, ties into 6.4)

`pdfinfo apresentacao.pdf` shows `Title` and `Author` populated (beamer
derives these from `\title`/`\author` automatically) but `Subject:` and
`Keywords:` are blank. Minor, but costs nothing to fix and helps anyone
who finds the PDF later via search/library metadata. Plan: add
`\hypersetup{pdfsubject={BitNets e redes neurais de pulso}, pdfkeywords=
{BitNet, spiking neural networks, quantização, STE, surrogate gradient}}`
to `preamble.tex` (same place as 6.4's fix, if both are done together).

### 6.6 One font subset embedded as Type 3 instead of Type 1

**File:** built PDF, root cause not yet located in the `.tex` sources

`pdffonts apresentacao.pdf` shows every font embedded as Type 1 *except*
one: `BMQQDV+DejaVuSans` is `Type 3`. Type 3 fonts in a pdflatex output
usually mean some specific glyph/shape got rasterized as vector paths by a
package (commonly a TikZ `\node` with certain text-rendering options, or a
symbol pulled from a font pdflatex couldn't subset normally) rather than
using the installed Type 1 DejaVuSans directly — Type 3 glyphs can look
visibly blurrier than Type 1 ones when a PDF viewer scales them, which
would show up as a some specific piece of text looking subtly worse than
the rest of a slide when projected large. **Not root-caused this
session** — `pdffonts` doesn't map object IDs to source locations. Plan:
bisect by commenting out slide `\input`s in `apresentacao.tex` (binary
search) and re-running `pdffonts` after each partial compile until the
Type 3 entry disappears, to identify which slide/macro introduces it;
likely candidate given the deck's structure is one of the TikZ diagrams
(`fundamentosArquitetura.tex`'s network diagram, or one of the
`\DemoSlide`/`\AtBeginSection` beamercolorbox templates, since those are
the main non-standard-text-rendering paths in the deck).

---

## Suggested execution order

1. **Item 4** (references contrast) — smallest, most isolated, no design
   decisions needed, do first.
2. **Item 2** (clipping regression test + `setMinimumSize`) — mechanical,
   adds a permanent safety net that item 1's rework should then be
   developed against.
3. **Item 1** (STE demo rework) — the biggest piece, benefits from item 2's
   clipping test already being in place to catch any new overflow from the
   added value-text in the boxes.
4. **Item 3** (Patrick image) — blocked on information only the user can
   provide; do the defensive try/except + regression test (plan steps 2-3)
   any time, but the actual root-cause fix waits on their diagnostic
   report.
5. **Item 5.1** (surrogate-gradient numbers) — do right after item 1, same
   pattern, same reviewer context still warm.
6. **Items 6.1/6.2** (nav symbols off, page counter) — trivial, do
   whenever touching `preamble.tex` next (item 6.4/6.5 touch the same
   file, bundle them).
7. Everything else in sections 5 and 6 (5.2, 5.3, 5.4, 5.5, 5.6, 6.3, 6.4,
   6.5, 6.6) — genuine improvements but lower value-per-effort than 1-4;
   pick up opportunistically, no dependency ordering between them.
