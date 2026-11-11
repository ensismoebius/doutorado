# Round 4 ruling — Section 5 conditions and the pre-validation question

Paper: `documentation/07-articlesProduced/meeting01/paper.tex`.
Basis: `dossier/round4_changes.md`, `dossier/ruling_math.md`, the C++ code, the
analysis scripts and the paper. I re-derived the Eq.(2) step, the t/p/Holm code,
the PCA energy arithmetic and the fold map. Nothing was built or run; the
harness, the main grid and the pre-validation remain unexecuted in this
snapshot. No paper file was edited.

Status vocabulary is the one requested: **SATISFIED**, **NOT SATISFIED**,
**CANNOT BE DONE BEFORE RUNS**. Where a condition has a form part and a result part,
the strict status is used and the satisfied part is named in the evidence
column.

## 1. Condition table

| Section-5 condition | Status | Evidence / residual |
|---|---|---|
| D1–D4 code: WAV data-chunk walk, 44-byte header, cap, window draw, 128→160 Hz, format 11, known-answer tests | **SATISFIED** | `Wav.cpp` walks `fmt `/`data`; `wave_gtest` covers header, extra chunk, no-data; `stratified_window_cap` and `cap_windows_per_recording` tests; `resample_bandlimited`; `kResultsFormat=11`; real headers checked in Python (FSDD 3000, AudioMNIST 30000 at byte 44; EEGMMIDB 1490 at 160 Hz + 36 at 128; Siena 512). |
| D2 coverage: per-fold composition printed/recorded | **SATISFIED** | `split_composition_json` writes speakers, digits and source-window indices into the manifest for train/val/test (`Meeting01Experiment.cpp` l.1522–1524); `split_audit_gtest` exercises the composition. |
| D1–D4 real C++ loader on FSDD/EEG | **CANNOT BE DONE BEFORE RUNS** | The tests are synthetic and the real header check is Python. The first real FSDD/EEG windows are the pre-validation's own split dumps; this is exactly what the pre-validation is for. |
| D1–D4 real C++ loader on AudioMNIST/Siena | **NOT SATISFIED** | Raw files have been read on the cluster, but no window has gone through the real loader. The main grid is blocked on a short split-dump job. |
| D5 PCA row and ratio | **SATISFIED** | `pca_energy_nj` gives 157.985 nJ (k=16) and 903.0 nJ (k=64); the table has PCA rows and SNN/PCA ratios; `_pca_checks` pins 4096 MACs, 256 mean adds, 272 address adds, 4096 weight reads at 13.33 pJ, 528 input/mean accesses at 10 pJ. Arithmetic is correct. |
| D5 PCA verdict | **CANNOT BE DONE BEFORE RUNS** | `paper_loso_energy.tex` is all `--` and `\energyVerdict` is “[the energy comparison has not run yet]”. The claim “PCA in the verdict” is not true yet; only the row and the rule are in. |
| D6 group-level primary | **SATISFIED** | `group_level` uses one `d_g` per held-out group, the Student-t interval, the one-sample t p, Holm within dataset and the forest; recording/seed/window levels are labelled descriptive. |
| D6 independence/effective n | **NOT SATISFIED** | See R4-3. The one-sample t treats the `d_g` as independent; the LOSO folds share training data and search, and the paper concedes only approximate normality. |
| D7 fresh-seed re-score | **SATISFIED** | `detail::rescore_front` re-scores the front on `base_seed + k·7919` (k=1..3), never the search seed; `RescoreAveragesFreshSeedsOnlyNeverTheSearchScore` pins the seeds and the mean. |
| D8 design/T1 regeneration | **CANNOT BE DONE BEFORE RUNS** | `paper_design.tex` is dashes and `tOneVerdict` is “[T1 has not run yet]”. This is a post-pre-validation job, not a pre-validation blocker. |
| D9 explicit Vth decision | **SATISFIED** | Eq.(1) says `R`,`C` are clamped and “nothing bounds V_th”; no Vth clamp in the code; `lif_v_th_min`/`lif_v_th_mean` are logged; the `Exact` comment is gone. |
| D9 physical Vth domain/range | **NOT SATISFIED** | See R4-5. Nothing keeps `V_th` positive; a negative threshold makes `1[V>V_th]` fire almost always. The trained range is a placeholder. |
| Verify AudioMNIST and Siena; do not start the grid on contaminated loaders | **NOT SATISFIED** | Same as the real-loader row; the pre-validation deliberately excludes both. |
| Fix or relabel the control (F3); drop “matched” | **SATISFIED** | Paper says the control “is not otherwise matched to the code, T or transform”; “matched” remains only in the figure caption's “identical widths, initializer and training”, which is true. F3 with real winners cannot be done before runs. |
| Exclude fold 1 / list development choices | **SATISFIED** | `DESIGN_INFORMED_FOLDS` = FSDD/EEG {1}; fold 1 is out of the headline; `DEVELOPMENT_FOLDS` = {0}; stopping level, cap, T range and surrogate sharpness are listed in §Experimental Setup. |
| Fold 5 and the “kept out” claim | **NOT SATISFIED** | See R4-2. `DESIGN_INFORMED_FOLDS` does not contain 5, yet fold 5's validation block is block 0, the design study's test block; the abstract and conclusion say the design folds and the folds that validated them are kept out. |
| Real main-grid numbers, no placeholder/TODO | **NOT SATISFIED** | Headline, significance, energy, pre-validation and T1 macros are placeholders; page limit is 7, so the 8-page limit itself is not the problem. |
| Page limit 8 | **SATISFIED** | Current PDF is 7 pages. |
| Group-level tests, per-subject effects, mean-based energy rule with margin | **SATISFIED** | `group_level`, the per-group table/forest and the two-axis rule are present; the margin's operating characteristics are R4-1. |
| Margin Δ = 0.02 operating characteristics | **NOT SATISFIED** | See R4-1. At n=4 the half-width is 1.59·s, so the interval fits ±0.02 only if s≤0.0126; the paper's 0.016 is the window-draw SD, a different component. |
| Energy table: 64-bit wording, encoder/decoder/mean/PCA, first-layer-once, absolute E | **NOT SATISFIED** | The form is in: §Cost states the 64-bit→32-bit overprice and the halved column; Table `tab:cost` has the first-layer-once stored-current row; `tab:energy` has PCA, per-side, SRAM-size, half-price and as-executed columns. The absolute values are dashes, so the “absolute energies corrected” part is still a grid result. |
| Eq.(2), reset detached, “exact” removed, Vth clamp text | **SATISFIED** | Eq.(2) is `σ̂'(V)⊙(Wᵀδ) + β(1−s)⊙δ_next`; `compute_grad_step` computes `grad_out*surr + grad_from_next*beta*(1−spike)`; “reset is detached” and the Vth surrogate asymmetry are stated; Vth is explicitly unbounded. |
| T1 described as plateau/null, numbers corrected | **SATISFIED** | The paragraph says the check “does not show that a Transformer cannot learn”; “0.935 vs 0.996” and “10 of 12” are gone; the verdict is a placeholder. |
| T1 result | **CANNOT BE DONE BEFORE RUNS** | Same row as D8. |
| Design table regenerated | **CANNOT BE DONE BEFORE RUNS** | Dashes until the logged twin reruns on the corrected windows. |
| “Leakage-controlled” / “development fold kept out” corrected; fold 1/5 disclosed; “matched” corrected | **NOT SATISFIED** | “Matched” is fixed and fold 1 is disclosed, but the abstract/conclusion “kept out” is false for fold 5 (R4-2). |

## 2. New real objections (round 4 only)

These are the points that are both new and stand on their own; the earlier
F1/MS/m codes are not repeated unless a round-4 change altered them.

**R4-1 (MAJOR) — Δ = 0.02 is not tied to the group-level test, and the rule is
unattainable on FSDD.** `inside_margin` is
`(-Δ < lo) and (hi < Δ)` on the 95% Student-t interval of the `d_g`
(`04_meeting01_significance_tests.py` l.266; `EQUIVALENCE_MARGIN = 0.02` in
`meeting01_results.py` l.300). At n=4 the half-width is
`t_{0.975,3}·s/√4 = 1.591·s`; the interval fits inside ±0.02 only when
`s ≤ 0.01257`. The paper justifies 0.02 as “2% of the no-model error and the
order of what a new window draw changes in itself (standard deviation 0.016 for
PCA on FSDD over 24 draws)”. That 0.016 is the draw-to-draw SD of the PCA error,
not the SD of the group-level `d_g`; the equivalence interval is over four
speakers, not over draws. The result is predictable: on FSDD the rule
cannot certify, so “energy first” can never recommend the lower-energy model on
that dataset, only report the trade-off. The test helper uses
`d = 0.001, 0.002, 0.003` (n=3), which hides the n=4 width. The paper's
limitation (i)/(x) says “wide by construction”, but that is a caveat, not a
derivation of Δ. Fix: either derive Δ from the `d_g` distribution (and state the
power), or drop the word “equivalence”, report the trade-off for every dataset,
and say the rule is a conservative screen, not a test. If the intent is a TOST,
the variance component used to size Δ must match the interval's.

**R4-2 (MAJOR) — Fold 5 validates on the design test block; the abstract and
conclusion's “kept out” is false.** `DEVELOPMENT_FOLDS = {0}` and
`DESIGN_INFORMED_FOLDS = {1}` for FSDD/EEG; fold 5's validation block is block 0,
and the design study tested on block 0. The paper's own figure caption says
“fold 5 stays, although it validates on the block the design study tested on”,
and §Development folds says the two datasets are design-informed. But the
abstract says the design study's folds are “excluded with the folds that
validated it from the headline numbers”, and the conclusion says “those folds
and the ones they validated on kept out of the headline numbers”. Both are
false for fold 5. Excluding fold 1 answers only half of V005: the fold that
*validated* on the design test block is still in the headline. Fix: either
exclude fold 5 as well (headline 2–4; FSDD n=3, EEG n=9) and regenerate the
tables, or rewrite the abstract/conclusion as “fold 1 is excluded; fold 5 and
all FSDD/EEG results are design-informed”. Disclosure is not exclusion.

**R4-3 (MAJOR) — The one-sample t-test treats the `d_g` as independent; LOSO
folds share training data and search.** `group_level` is a one-sample t on four
FSDD or twelve EEG `d_g`, with `se = std(d, ddof=1)/√n` and
`t.ppf(0.975, n−1)`. A fold's model is trained on the other groups and each
fold is searched separately; the `d_g` therefore share training samples, the
same architecture family and the same design choices, and are not iid. The
paper concedes only that “the t interval rests on approximate normality, which
4 points cannot check”; it does not state the independence assumption, which is
the one the interval actually needs. At n=4 the effective n is below 4, so the
reported p and interval can be anti-conservative or conservative by an unknown
amount, and the Holm correction inherits the same problem. Fix: state the
dependence explicitly, replace or supplement the t with a group-level
permutation/bootstrap that respects the fold structure (or report the interval
as descriptive only), and stop calling the Holm-corrected p a test of the
group-level effect until that is done. The direction of the bias is dataset
dependent; a caveat that does not name it is not enough.

**R4-4 (MAJOR, reporting) — The EEGMMIDB headline n is 12, not 109; the
limitation conflates the dataset count with the headline test count.** With
groups of 3+3+3+3 for folds 2–5, the headline `n_groups` for EEGMMIDB is 12.
The limitation says “subjects of the headline folds of 60, 109 and 14 groups”.
The 109 is the full EEGMMIDB subject count (`#G` in Table `tab:datasets`), not
the headline test count; AudioMNIST's 60 and Siena's 14 are whole-dataset counts
because those datasets use folds 0–5, and FSDD's 4 is the headline count. The
sentence therefore overstates the EEG test sample by 97 subjects. Fix: report
the actual test counts (FSDD 4, EEGMMIDB 12 of 109, AudioMNIST 60, Siena 14),
and make the table's `n_groups` column agree. This matters because the t and
Holm are over 12, not 109; the precision is that of 12.

**R4-5 (MAJOR) — An unbounded `V_th` can become negative, which is not a
physical LIF threshold, and the trained range is unverifiable.** Eq.(1) says
“nothing bounds `V_th`”; `compute_grad_step` reads `threshold_val` directly
from the trained parameter; the spike is `1[v_pre > threshold_val]`. A negative
threshold makes the neuron fire almost always, which is not the LIF regime and
inflates the spike-event energy. The
paper's “its trained values are in Table~\ref{tab:snnsel}” is a placeholder, so
the one check that would reveal this cannot be made. The project rule against
silent clamping is right; the answer is not an unconstrained physical
parameter. Fix: either parameterise `V_th` with a positive map (softplus,
exp, or a hard lower bound) and retrain, or keep the unconstrained value but
report the minimum/mean per layer and state that a negative value means the
“spiking” model is outside the LIF reading. If a positive bound is adopted, the
pre-validation and the main grid must be rerun; this is a training change, not
a text change.

**R4-6 (MINOR, pre-validation risk) — The real C++ loader has not been
exercised on real FSDD/EEG, so a 13 h pre-validation can be wasted.** The
`Wav.cpp` fix, the format-11 bump and the header checks are all in, and the
gtest suite is a real test of the parser, but every test file is synthetic. The
Python check at byte 44 proves the *files*, not the C++ path; the EEG and
AudioMNIST readers are likewise not driven by real windows here. The
pre-validation is the first real FSDD/EEG run. Fix: before the 13 h job, run a
short split-dump smoke (one fold, one run, few epochs, write the train/val/test
composition, the first/last window values and the per-window energy); inspect
it; only then launch. This is a sequencing condition, not a paper objection.

**R4-7 (MINOR) — The margin is provisional and the paper says “we fixed”.**
`round4_changes.md` says “Provisional: the paper's author has not confirmed it
yet”; the paper says the rule was “fixed, after seeing the development folds
and before the main run”. Both can be true only if the author confirms and
dates the choice before the grid. As written, the margin is an adaptive choice
made after the development folds and is not pre-registered (m28). Fix: have the author
confirm the value and the date, say “chosen provisionally before the main grid”
rather than “fixed”, and keep the sensitivity of the energy verdict to Δ
visible. This is a small honesty point, but it is the same class as the
pre-registration issue the paper already concedes.

**R4-8 (MINOR, condition accounting) — “PCA in the verdict” is not yet true.**
The arithmetic and the row are right, but `paper_loso_energy.tex` is dashes and
`\energyVerdict` is a placeholder. The D5 claim in the change note is therefore
ahead of the paper. Do not mark D5 complete until the grid writes the verdict;
the axis rule and the PCA row can stay as they are. This is not a new
mathematical error; it is a status correction so the pre-validation is not
mistaken for a D5 result.

## 3. Pre-validation verdict

**YES — the format-11 pre-validation on the development folds may start now,
separately from the main grid, under conditions.** It is a development-fold
training/plumbing run, not evidence: FSDD and EEGMMIDB fold 0 only, no
AudioMNIST, no Siena, no headline numbers. The reasons are:

- The code-level conditions that affect a training run are in: the WAV
  data-chunk walk and format bump, the cap/draw/128→160-Hz fixes, format 11,
  D7's fresh-seed re-score and D9's explicit Vth decision. The tests are
  synthetic, but they exercise the branches the pre-validation will use.
- The open analysis objections (R4-1, R4-2, R4-3, R4-4) do not change what the
  pre-validation trains or measures; they change how the headline is computed
  and reported. The pre-validation can produce the split dumps and the Vth
  range that those fixes need.
- The open loader condition is AudioMNIST/Siena, which the pre-validation
  excludes. D8/T1 and the PCA verdict are post-pre-validation jobs; they do
  not block it.

Conditions:

1. **Smoke first (R4-6).** A short split-dump run (one fold, one run, a few
   epochs) through the real C++ loader, with the train/val/test composition,
   first/last window values and per-window energy written out. If it is clean,
   launch the ~13 h job; if not, fix and rerun the smoke. The header check is
   not a loader run.
2. **Development only.** Label the output `prevalidation`, never `headline`;
   do not promote it to the main tables; do not pool it with the grid.
3. **Vth range first (R4-5).** Log `lif_v_th_min`/`lif_v_th_mean` per layer and
   inspect them before the main grid. If `V_th` goes negative, either adopt a
   positive parameterisation and rerun the pre-validation, or keep it and write
   the R4-5 caveat; do not let the main grid start on an unexamined negative
   threshold.
4. **No AudioMNIST/Siena until their split dumps are checked.** The
   pre-validation does not cover them; the main grid does.
5. **Do not use the pre-validation to settle** the equivalence rule, the fold-5
   exclusion, the t independence or the PCA verdict. Those are analysis/report
   fixes (R4-1 to R4-4, R4-8) and must be made on the main-grid output.

**Main grid: NO, not now.** It waits for (a) a clean pre-validation, (b) the
AudioMNIST/Siena real split dumps, (c) the Vth-range decision, and (d) the
R4-1/R4-2/R4-3/R4-4 reporting fixes so the headline is computed the way the
paper says. The first three are sequencing/data-path conditions; the fourth is
a paper/analysis condition, not a reason to retrain. If a positive Vth bound is
adopted, the main grid waits for that retrain as well.

**What would change this verdict to NO:** a smoke that shows wrong windows,
wrong composition or wrong energy; a negative Vth range that the author wants
to bound; or a decision to change the training/cap/stopping design before the
grid. In those cases the 13 h pre-validation would be measuring a pipeline that
is about to change.

## 4. Limits of this ruling

- No build, no GPU, no cluster, no real data: every code claim here is a read of
  the source and tests, not an execution. The pre-validation is precisely the
  execution this ruling cannot do.
- The headline, significance, energy and pre-validation numbers are
  placeholders; no effect size, p, interval or energy verdict is verified.
- The PCA arithmetic is checked by hand and by the in-file `_pca_checks`; it is
  not a chip measurement.
- The fold-5 and EEG-n=12 claims are derived from `assign_speaker_fold` and the
  fold map, not from a run's manifest. The pre-validation's split dumps should
  confirm them.
- The t/Holm and equivalence comments are statistical, not numerical: they
  concern the assumptions the code's formulas rest on, not a coding error in
  `t.sf` or `_holm`.

## 5. One-line answer

**Start the format-11 pre-validation now, but only as a development-fold
plumbing run with a real-loader smoke first and no headline status; the main
grid stays closed until the pre-validation, the AudioMNIST/Siena loader dumps
and the Vth decision are in, and the paper must fix the fold-5 “kept out” claim,
the n=12 EEG count, the Δ/independence mismatch and the negative-Vth
possibility before any headline result is credible.**

