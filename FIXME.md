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
