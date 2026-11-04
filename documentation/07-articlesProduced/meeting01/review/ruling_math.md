# Ruling on the Author's response (Reviewer 2, round 3)

Paper: `documentation/07-articlesProduced/meeting01/paper.tex`.
Basis: round-2 `dossier/review_math.md`; `dossier/author_response.md`;
`dossier/response_scripts/`; the C++ code; the paper. I reproduced the M9, M13,
M4 and M5 numbers, the ICC, and the split/fold mapping. The headline grid is
still placeholder (`paper_loso_*`, `paper_prevalidation*`), so F1 stands.

Evidence grade of the response: numpy/plain-Python replica, 1-2 seeds, one fold,
not harness output. That is adequate to settle algebra and code paths, not to
settle every effect size.

## 1. The 26 review objections

| Objection | Status | One-line reason |
|---|---|---|
| F1 headline absent | SUSTAINED | Main-grid macros are placeholders; no SNN-AE vs ANN-AE number exists. |
| F2 saturation "contradicted" | MODIFIED | Trained sigmoid is not saturated (0.3% pinned, EEG); the design table does not contradict saturation; the paper's mechanism sentence is still not earned, and "clamp" vs "staircase" is imprecise. |
| F3 control not matched | SUSTAINED | Author concedes (V004): encoding, T, transform and decoder differ; "matched" is false. |
| M4 Eq.(2)/code/exact | MODIFIED | Core stands: Eq.(2) gives 1.5 at both v, code 0.5/1.5, "Exact local derivative" is false. My paper column (0.5/0.5), the "exact" column and the finite-difference demand are withdrawn: the paper is 1.5/1.5; "exact" is one surrogate choice; finite difference is ill-posed. |
| M5 64-bit energy on 32-bit words | MODIFIED | Disclosure/absolute-E error stands (ANN 854.5 vs 501.5 nJ at 64/32-bit). My 1.704 is the ANN's own 64/32 ratio, not the SNN/ANN effect; the SNN/ANN ratio moves only 1-12% and never crosses 1. |
| M6 first-layer "against the conclusion" | MODIFIED | The blanket sentence is false (author concedes C151); "not disclosed" is withdrawn: line 287 and the abstract disclose the repeat. |
| M7 first-layer-once row | SUSTAINED | The table lacks the row; prose alone is not enough. |
| M8 table omits encoder/mean/nonlinearity | MODIFIED | Mean note is wrong and the table under-describes; "undercharges the SNN" is withdrawn (the code charges them). |
| M9 re-score keeps search seed | MODIFIED | Code defect is real (D7). My "29% bias" and fresh-only ≈0 are artifacts: `checks.py` draws fresh scores without mu. Corrected: selection quality unchanged, reported validation score about twice as biased, test unaffected. |
| M10 winner's curse sign | WITHDRAWN | Sign reversed: lower is better; the selected family at -1.58 sd wins. "Conservative" still not a theorem, and the author removes it (P05) for the control reason, not the sign. |
| M12 initializer claim | SUSTAINED | FSDD T=4 Kaiming 0.724 vs Xavier 0.729: "better or equal for each arm on both datasets" is false. |
| M13 pseudoreplication | SUSTAINED | Empirical ICC 0.33, design effect 5.3, 266 recordings/19 subjects; group-level 19/19 positive, d=0.154, cluster SE 0.0044. 0.33/5.3 are a one-fold point estimate, not a population parameter. |
| M14 "Energy first" rule | SUSTAINED | Author concedes C206; rule needs a mean-based estimand and margin. |
| m15 T1 underpowered | SUSTAINED | V011: 10-11 of 12 FSDD fits at the mean predictor; "0.935 vs 0.996" is void. |
| m16 "every model" / PCA energy | SUSTAINED | PCA has no energy row; PCA k=16 (158 nJ, 0.649) can dominate ANN (854 nJ, 0.667). |
| M17 depth range | SUSTAINED | Author concedes C120/P11. |
| M18 design confounded | SUSTAINED | One-factor-at-a-time, cells confound loss/decoder/init/activation; regenerate with D8. |
| M19 per-step/decoder not separable | MODIFIED | The table does separate them sequentially; "not separable" is withdrawn. New: decoder changes error by -0.005 (FSDD) and +0.003 (EEG), so "a spiking decoder added error" and the abstract's "two avoidable choices" are unsupported for the decoder; its effect is cost. |
| m20 Vth unclamped | SUSTAINED | Eq.(1) clamps Vth, code does not; "fix the text" is not a decision. |
| m22 "48 evaluations" | SUSTAINED | Nominal; say "48 candidates" or report distinct trainings. |
| m23 profile bounds | WITHDRAWN | Profile exists; Vth 0.5-1.5, alpha 0.8-0.99, T 2-32, pop 8, gen 5, seeds 3 match the macros. |
| m24 "below 8 KB" | SUSTAINED | Wording: "at or below". |
| m25 threshold-add citation | WITHDRAWN | Paper says only that Lemaire do not count the threshold test; no missing citation. |
| m26 stale test rows | WITHDRAWN | Restore removed; every run recomputes; no stale row can reach a table. |
| m27 "development fold kept out" | SUSTAINED | Vacuous for AudioMNIST/Siena; say the design study is FSDD/EEG only. |
| m28 "fixed in advance" | SUSTAINED | No pre-registration; use "after seeing the development folds". |

Count: 15 sustained, 7 modified, 4 withdrawn.

## 2. The pipeline defects D1-D9

| Item | Ruling | Reason / condition |
|---|---|---|
| D1 WAV header | ADEQUATE (conditional) | Real bug; all 3000 FSDD files data at byte 44. `seekg(sizeof(header))` plus a data-chunk check, known-answer test and format bump is safe-fail for FSDD. AudioMNIST is unverified: check the real sox header or drop AudioMNIST. |
| D2 cap truncation | ADEQUATE (conditional) | FSDD mild; EEG validation 11/18 subjects; AudioMNIST port trains on 2 speakers and tests on 3. The seeded, group/digit-stratified draw plus a coverage test is the right fix, but the AudioMNIST port is not evidence; verify on real files or drop. |
| D3 first windows | ADEQUATE | Seeded sampling across the recording; applies to AudioMNIST/Siena. |
| D4 128 Hz EDFs | ADEQUATE | Exclude S088/S092/S100 (or per-file rates); 42/1000 training windows, test blocks 4/5. Must land before the grid. |
| D5 PCA energy | ADEQUATE | Add a PCA row; this can change the conclusion, not just the table. |
| D6 reporting | PARTIAL | Group-level primary is right, but a fold-1-out sensitivity is too weak: fold 1's test is fold 0's validation block (direct contamination). Exclude fold 1 from the primary or label it design-informed; FSDD then has 4 speakers, min exact p 0.125. |
| D7 re-score seed | ADEQUATE | One-line fix; no test result changes. |
| D8 stale inputs | ADEQUATE (conditional) | Regenerate the design table after D1-D4 and rerun T1; the hand-typed format-7/8/9 table is not current. |
| D9 small | PARTIAL | "Fix the text" for Vth is ambiguous: clamp Vth in code or remove it from Eq.(1) and say why. "Exact" comment must go; the `dvpost_dVth` (surrogate-through-reset) vs `dvpost_dvpre` (no reset) inconsistency needs a decision, not just a comment. |

## 3. The three REBUTs

- **m23 WITHDRAWN.** The profile exists and the macros match.
- **m25 WITHDRAWN.** The paper does not attribute the add to Hoefflin; it says
  only that Lemaire do not count it, which is true. No citation is missing.
- **m26 WITHDRAWN.** Removing the restore means every run recomputes its rows;
  the stale-row risk is gone.

## 4. New / remaining objections

1. **V005 fold-1 contamination (SUSTAINED).** `assign_speaker_fold` gives fold f
   test group f, validation group (f+1) mod 6. Block 1 is fold 0's validation
   block and fold 1's test block; block 0 is fold 5's validation block. Fold 1
   is directly design-informed, so "leakage-controlled" and "development fold
   kept out" are false, and a fold-1-out *sensitivity* does not repair the
   primary. Drop fold 1 from the primary or label it design-informed; list the
   undisclosed development choices (stop skill 0.10, 1000 windows, T range,
   surrogate sharpness).
2. **D5 PCA baseline (SUSTAINED).** PCA k=16 at 158 nJ / 0.649 is cheaper and
   better than ANN [59,16] at 854 nJ / 0.667; "Energy first" must include it.
3. **D9 Vth ambiguity (SUSTAINED).** "Fix the text" is not a decision; clamp
   Vth or remove it from Eq.(1) explicitly.
4. **M19 decoder claim (SUSTAINED).** Decoder effect is -0.005/+0.003; the
   decoder choice is a cost choice, not an error choice; fix the abstract and
   design finding (i).
5. **Job 5316016 (MODIFIED).** EEG is decision-grade, not printable "as is":
   folds 4/5 test S088/S092/S100 at the wrong rate, validation covers 11/18
   subjects, test is the first 64 s. T1 fold 0 is unaffected, so the go/no-go
   use is fine. FSDD is not usable: filter windows >= 1 for the decision, do not
   print it.
6. **D1/D2 AudioMNIST (SUSTAINED).** The real AudioMNIST header/cap is
   unverified; the paper's AudioMNIST and Siena rows should be dropped until
   verified.
7. **D2/D4 headline mapping (SUSTAINED).** D4 hits headline folds 4/5; D2 hits
   EEG validation in every fold; both must be fixed before the headline grid.
8. **F2 staircase terminology (MODIFIED).** The LIF transfer with direct input
   and hard reset is a staircase, not a clip; the sigmoid is not saturated; the
   saturation mechanism is a hypothesis, not a measured result.
9. **Q3 same-init vs cross-init (MODIFIED).** "+0.008" is same-init sigmoid vs
   leaky ReLU; the design table's 0.308 vs 0.489 crosses initializers. Say
   which comparison each sentence uses.
10. **ICC model-vs-training (MODIFIED).** The 0.33 is the LIF-vs-ANN
    model-difference ICC; same-model A/B training differences give ICC ~0 (both
    on a plateau). The 0.33 is the relevant diagnostic for the model contrast,
    but do not describe it as two trainings.
11. **Design table and T1 stale (SUSTAINED).** D8: regenerate after D1-D4;
    replace the format-7/8/9 hand-typed cells; rerun T1 on current winners.
12. **T1 plateau (SUSTAINED).** 10-11/12 FSDD fits at the mean predictor;
    "0.935 vs 0.996" must go; Q14's "10 of 12" must agree with V011's 11 fits
    near the mean (state the marginal 0.990 fit).

## 5. Conditions

Before the main grid:
- D1-D4 land and are verified on real FSDD and AudioMNIST/EEG headers; format
  bump and known-answer tests; D2 coverage test prints per-fold composition.
- D5 PCA row, D6 group-level primary, D7 fresh-seed re-score, D8 design/T1
  regeneration, D9 explicit Vth decision.
- Drop or verify AudioMNIST and Siena; do not start the grid on D1-D4-contaminated
  loaders.
- Fix or relabel the control (F3); drop "matched" until encoding/T/transform are
  matched.
- Exclude fold 1 from the primary or label it design-informed; list development
  choices.

Before submission:
- Real main-grid numbers replace every placeholder; no `TODO`/placeholder macro
  remains; page limit 8.
- Group-level tests primary, per-subject effects printed; mean-based "Energy
  first" estimand with margin.
- Energy table/prose: 64-bit vs 32-bit wording, encoder/decoder/mean/PCA rows,
  first-layer-once row; absolute energies corrected.
- Eq.(2) matches the code or states "reset detached"; "exact" removed; Vth
  clamp explicit.
- T1 described as a plateau/null result, not learning; "0.935 vs 0.996" and
  "10 of 12" corrected.
- Design table regenerated; initializer and decoder claims corrected; abstract's
  "two avoidable choices" corrected for the decoder.
- "Leakage-controlled" and "development fold kept out" corrected; fold 1/5
  disclosed; "matched control" corrected.

## 6. Final verdict

**REJECT (current submission), with a credible path to MAJOR REVISION.**
The response is substantially honest: 33 of 48 C-codes are conceded and the
D1-D9 plan is mostly right. But the paper still has no headline result, its only
real table (T1) is at the mean predictor on FSDD, its control is not matched,
its loaders carry D1-D4, its protocol has a direct fold-1 leak, and its energy
and design claims are wrong in the direction that flatters the SNN-AE. The
current submission cannot be accepted. Resubmit after the main grid and every
condition above.

## 7. Where I disagree with the Author

- **D1:** do not call it closed while AudioMNIST is unverified; `seekg(sizeof(header))`
  is only safe because of the added data-chunk check, and that check must be tested.
- **D2:** the AudioMNIST port is a code simulation, not evidence; the real files
  must decide the fatal/non-fatal question.
- **D4:** "Use EEGMMIDB as is" is too generous for the headline; folds 4/5 need
  the three subjects excluded. T1 fold 0 is the exception.
- **D6:** a fold-1-out sensitivity is not enough; fold 1 is directly
  design-informed. Make the exclusion primary or relabel the headline.
- **D9:** "fix the text" for Vth leaves two incompatible statements; choose clamp
  or remove, and decide the `dvpost_dVth`/`dvpost_dvpre` asymmetry.
- **Job 5316016:** "usable with disclosure" for EEG is decision-grade only; not
  printable as is.

## 8. Limits of this ruling

- The main grid/pre-validation are placeholders; I cannot verify any headline,
  significance, energy-table or pre-validation number.
- The design study is a PyTorch replica; the C++ harness was not built here.
- Siena/AudioMNIST data are absent; D2/D3 for them rest on code and a port.
- The ICC 0.33/5.3 is one fold, one model pair, 266 recordings; it is a
  diagnostic, not a population estimate.
- The author's v001/v002/v006/v011 scripts were run where present; their exact
  tables are otherwise author-reported.
