# Reviewer 1 ruling, Round 4 (2026-11-05)

Read: `round4_changes.md`, `paper.tex`, `paper.pdf` (8 pages), the cited code. Python self-tests of `00_` to `06_` pass here. I did not build or run the C++ tests. Nothing below is a result.

**Verdict on the question: YES, the format-11 pre-validation may start now**, separately from the main grid, under the four conditions in Section 3. The main grid may not start.

## 1. Conditions

### My Round 3 conditions before the grid

| # | Condition | Status | Note |
|---|---|---|---|
| 1 | D1 to D4 | SATISFIED | Chunk walker, balanced cap, uniform draw, per-file rate. C++ not run by me |
| 2 | Tests | SATISFIED in code | Tests exist. No C++ pass is reported, so it is a start condition |
| 3 | AudioMNIST, Siena windows seen | NOT SATISFIED | Raw files only. Not a pre-validation blocker: neither is in that job |
| 4 | s recalibrated (N1) | SATISFIED | 84 runs, 29 plateau, peak 0.0505 recomputed from the CSV. Paper wording: R6 |
| 5 | Checkpoints, format | SATISFIED | `checkpoint_hash` covers every value of all splits; format 11 |
| 6a | Margin Δ fixed | NOT SATISFIED | Provisional, and its rationale fails (R2) |
| 6b | Group-level primary | NOT SATISFIED | Done for subjects, but the wrong unit on three datasets (R1) |
| 6c | Encoder column, PCA energy | SATISFIED | PCA 158.0 and 902.7 nJ recomputed |
| 7 | Window draw decision, D7 | SATISFIED | Fixed per fold, disclosed; fresh seeds confirmed |
| 8 | Logging | SATISFIED | Shares have a reading limit (R3) |
| 9 | Design study and T1 | SATISFIED by removal | Regeneration: CANNOT BE DONE BEFORE RUNS |

### My Round 3 conditions before submission

| # | Condition | Status | Note |
|---|---|---|---|
| 1 | Saturation sentence gone | SATISFIED | One remnant at line 632 (R4) |
| 2 | Logged twin, ≥4 seeds, crossed, Poisson | CANNOT BE DONE BEFORE RUNS | |
| 3a | Text fixes | NOT SATISFIED | R4, R5, R6, line 53 |
| 3b | Figures | NOT SATISFIED | F1 left and F3's curve are maths and can go in now. F2, F7, m8 are in. F1 right, F3 winners, F5, F6: CANNOT BE DONE BEFORE RUNS |
| 3c | Placeholders | CANNOT BE DONE BEFORE RUNS | |
| 4 | Pre-validation rerun | CANNOT BE DONE BEFORE RUNS | This job |
| 5 | Design-informed label | SATISFIED | Lines 390 to 393 |
| 6 | D1 disclosure | SATISFIED | Lines 431 to 435 |

### N1 to N11

| N | Status | Note |
|---|---|---|
| N1 | SATISFIED | See R6 |
| N2 | SATISFIED | Files resampled, not excluded; 109 subjects keep their blocks |
| N3 | SATISFIED | Decoder claim removed; table must show the one-factor contrast |
| N4 | NOT SATISFIED | Only line 632 is left |
| N5 | SATISFIED | Bounds removed; lines 635 to 638 hold |
| N6 | SATISFIED | Line 403 is correct, `t` used |
| N7 | SATISFIED | Line 393 |
| N8 | NOT SATISFIED | Line 53 still says "energy is the cost we put first", while the search ignores E and dev gaps sit far outside Δ |
| N9 | SATISFIED | Claim removed |
| N10 | SATISFIED | Fixed draw, so no reference changes; but limitation (x) overclaims (R2) |
| N11 | SATISFIED | By reading; not exercised |

## 2. New objections

**R1 · MAJOR · The group-level test is pseudoreplicated at fold level.** `group_level` keys on (fold, subject), so EEGMMIDB has n ≈ 72 subjects from only 4 folds. Every subject in a fold shares the same selected architectures and trained models, so a fold effect is one sampled value repeated 18 times. Replica subject SD 0.019 gives an interval half-width near 0.005 at n=72. A between-fold SD of 0.01 alone gives 0.016 on 4 fold means. A subject-level pass of ±0.02 could therefore be false. Winners differ by fold, so this is the expected case. AudioMNIST and Siena have the same problem. On FSDD fold and group coincide, but training sets overlap across folds, so d_g are not independent there either; line 405 names normality only. *Fix before the grid, in `04_` only:* add a fold-means interval, require both for the Δ rule, state the overlap, mark folds in the forest plot. No run is needed.

**R2 · MAJOR · Δ = 0.02 is usable as a number, but its stated reason is wrong.**
- 0.016 is the SD of PCA's *level*, unpaired, on FSDD, one fold. It came from my own unreleased script, on selection that format 11 changed. Δ applies to a *paired* gap, whose draw spread was about 0.006 on the Author's three draws.
- Noise does not define irrelevance. A fixed margin does not widen an interval, so "which Δ is sized to" in limitation (x) is false.
- 0.02 means different things: 0.12 dB at FSDD's 0.71, 0.29 dB at EEG ANN's 0.29, 0.46 dB at EEG PCA's 0.18.
- Defensible form: "2% of the window's variance, a change we treat as negligible", with the dB range stated, and no reference to draw noise. Confirm it, or a replacement, **before anyone reads format-11 test numbers**. Note that dev gaps of 0.06 to 0.15 mean the branch will probably not fire for SNN against ANN; the paper should not imply that it decides much.

**R3 · MINOR · Per-side shares are not comparable across T.** With T up to 32, silent means 0 spikes and always means T spikes. At T=2 either is easy, at T=32 hard, and Table VIII mixes winners. Read them only with T beside them, or per T band. There is no control row, though the replica's leaky ReLU had 49% of values at ≤ 0.01 on FSDD. Add one before the grid.

**R4 · MINOR · Leftover sentences.**
- Line 632-633: "a hidden code near 0 and 1 is both" asserts a state the paper does not show. Say "if".
- Lines 81-83: "measured on the *same* network", "the spike adds", against contribution 1.
- Line 649: limitation heading "Matched, not identical inputs".
- Line 249: "every network starts on that plateau".
- Line 204: the recurrent cell's soft reset, `>=`, and untrained genes differ from "the network's".
- Line 155: Cawley still carries "group-disjoint".

**R5 · MINOR · Claims with their data removed.** "The better of the two we tried" (452), "These networks" (457), "the gap that remains" (453, 670) all presuppose results that are now dashes. Hold them until the runs, then keep only what the data show.

**R6 · MINOR · Line 252.** "Calibrated" hides that the script's pre-set rule refused 0.10 and the author kept it. The replay makes this immaterial (13 of 2349 epochs), so one clause fixes it.

**R7 · MINOR · Space.** Page 8 has roughly a third of a column free. F1 left, F3 and F5 need about 12 cm, so check the fit before the runs.

**Tested and found sound**
- **Fold-1 exclusion:** `is_headline_fold` filters 02 and 04 throughout. It is the right call and is disclosed, including fold 5. Dropping fold 5 too would leave n=3, so I do not ask for it.
- **t at n=4:** acceptable as disclosed. Critical t is 3.18, and 4.18 for ANN-AE once Mean takes the first Holm step. Equivalence needs group SD below 0.0126, so FSDD will rarely pass, which is honest. Dropping Mean from the family would help.
- **PCA price:** correct, and consistent with the networks' reads, with no hidden favour.
- **Fresh-seed re-score:** correct. Seeds differ from the search seed and from other runs' seeds, and the search score is excluded.

## 3. Pre-validation: conditions of start

1. Commit first, so that provenance is clean, not `+dirty`.
2. Full C++ meeting01 and core tests plus Python self-tests green on that commit, plus the existing smoke job on the cluster build.
3. Before reading its test numbers: confirm Δ (R2) and write down the fold-level rule (R1). This does not delay the start. It stops the rule being chosen after another look at the gaps.
4. If a control share row (R3) is wanted from this run, add it before deploying; otherwise it waits for the grid.

Not blockers: AudioMNIST and Siena (not in this job), T1 and the twin (they follow it), and all text items. Compute is not a concern: the 13 h estimate sits well inside the 5-day limit.

The main grid stays held until R1, R2, condition 3 (AudioMNIST and Siena windows, training splits preferred), and the user's confirmation.

## 4. Could not verify

C++ build and tests; `resample_bandlimited` behaviour beyond its test's existence; AudioMNIST and Siena windows; the 13 h estimate; actual between-fold spread (the 0.01 in R1 is illustrative, not measured).
