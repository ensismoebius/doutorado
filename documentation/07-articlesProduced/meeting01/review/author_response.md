# Author's response to the two hostile reviews

Round 2. Answers `review_visual.md` (V, Reviewer 1) and `review_math.md` (F/M/m, Reviewer 2).
Part A, the pipeline defects, comes first because it has priority. Parts B to E follow.

**Tally (48 objections): 33 CONCEDE, 12 PARTIAL, 3 REBUT.**
CONCEDE means the objection stands, possibly with corrected numbers. PARTIAL means a material part of it fails.

**How this was checked.** No build, no paper edit, no commit, no ssh, no GPU. Every run is in `dossier/response_scripts/`: numpy or plain Python, the longest about two minutes (13 short trainings of the reviewer's replica, `clean_probe.py`).

Read-only access outside the snapshot, as allowed: the saved `*_windows.npy` and per-window CSVs under the real repo's `results/meeting01_t1/`, and the FSDD and EEGMMIDB databases. No `.env` file was opened. No hook blocked a command this round. One of my shell lines hung on an empty variable and I stopped it; it had run nothing else.

**Evidence grade.** Replica numbers are direction, not harness output: reviewer's numpy twin, 1 or 2 seeds, direct input, one hidden layer. Everything marked "reproduced" uses the harness's own saved windows.

---

## Part A. Defects in code or data pipeline

Order is the order of fixing. D1 to D4 must land before the main grid.

| ID | Defect | T1 | Job 5316016 | Main grid | Design study |
|---|---|---|---|---|---|
| D1 | WAV header read as audio | FSDD meaning invalid | FSDD affected | Blocker | FSDD column invalid |
| D2 | Window caps truncate by file order | FSDD and EEG, mild | Both, mild | Blocker (AudioMNIST fatal) | Same as T1 |
| D3 | Per-recording cap keeps first windows | EEG, disclosure | EEG, disclosure | Blocker for AudioMNIST, Siena | EEG, disclosure |
| D4 | 128 Hz EDFs processed as 160 Hz | EEG, small | EEG, small | Fix | EEG, small |
| D5 | PCA has no energy | n/a | n/a | "Energy first" not computable | n/a |
| D6 | Group-level results not printed; no fold-1 sensitivity | n/a | n/a | Fix before analysis | n/a |
| D7 | Re-score keeps the search seed | Reported val only | Reported val only | One-line fix | n/a |
| D8 | Twin and T1 inputs are stale | T1 rerun | n/a | n/a | Regenerate |
| D9 | V_th unclamped; "exact" comment; stale comments | none | none | Tidy | none |

### D1. WAV header fed to the networks (V001). CONFIRMED, measured

**Path, read in full.** `load_grouped_windows` (`Meeting01Dataset.cpp`) sends both `"fsdd"` and `"audiomnist"` to `FsddWindowDataset`. That calls `FsddLoader::load_signal`, then `Wav::read` (`Wav.cpp`), which runs `read_wave_headers` and then `read_wave_data`. `read_wave_headers` ends at `Wav.cpp:373` with `ifs.seekg(0, std::ios::beg)` after measuring the file size. `read16BitMono` then reads `amountOfData` samples from the current position. Nothing seeks in between.

**Picture.**

```
file bytes:   [44-byte header][s0 s1 s2 ... s(n-1)]
read samples: [h0 ... h21][s0 s1 ... s(n-23)]        n values, last 22 real samples lost
```

**Measured (`v001.py`).** All 3000 FSDD files have the data chunk at byte 44, 16-bit mono, 8 kHz.

| Quantity | Result |
|---|---|
| Saved windows vs rebuild with header | max difference 1.3e-5 on all 2650 windows: exact |
| Saved windows vs rebuild without header | RMSE about 1.4 z-units, max 11.4 |
| Window 0 share | 8.2% of train, 7.8% of test |
| Energy of window 0 in its first 22 samples | 86% |
| Peak of a z-scored window 0, mean | 6.8 |

| Test MSE, FSDD fold 0 | With header | Clean | With header, window 0 | With header, other windows | Clean, other windows |
|---|---|---|---|---|---|
| Mean predictor | 0.9954 | 1.0008 | 0.862 | 1.0067 | 1.0008 |
| PCA k=16 | 0.6492 | 0.7074 | 0.095 | 0.6962 | 0.6991 |
| PCA k=64 | 0.2072 | 0.2425 | 0.041 | 0.2213 | 0.2230 |

V001's 0.649 to 0.707 and 0.9955 to 1.0008 are reproduced.

**What it does to the gap.** Less than the reviewer's one seed suggested.

| Replica, Xavier, FSDD | ANN-AE | SNN T=4 | Gap | Seeds |
|---|---|---|---|---|
| With header (reviewer) | 0.667 | 0.727 | 0.060 | 2 |
| Clean (reviewer) | 0.710 | 0.792 | 0.082 | 1 |
| Clean (mine, `clean_probe.py`) | 0.7153 | 0.7793 | 0.064 | 2 |

The header moves absolute error by about 0.05 and the gap by about 0.004. The reviewer's 0.082 was a high draw: my two seeds give 0.773 and 0.785. His reading that the clean ANN-AE sits at PCA holds: 0.715 against 0.707.

**A salvage the reviewer did not offer.** Windows with `source_window_index >= 1` are real speech, only 22 samples late. Filtering them out of a contaminated run gives nearly the clean number:

| | Header run, other windows | Clean run, all windows |
|---|---|---|
| PCA k=16 | 0.6962 | 0.7074 (clean other windows 0.6991) |
| ANN-AE replica | 0.714 | 0.715 |
| SNN T=4 replica | 0.778 | 0.779 |

Those last two rows compare different sets of windows, so read them as approximate. They are close enough for a go/no-go decision on the job, not close enough to publish.

**Other consequences.**
- *Armed patience.* The stopping rule arms at 10% skill against the mean predictor. On FSDD, 8% of windows at error near 0.11 give about 6% skill with nothing learned about speech. A model needs only about 4% real skill to arm. The rule did not mean on speech what it was designed to mean.
- *Search.* Validation contains window 0 too (about 4% in the saved fold), so GA selection saw it.
- *Tests.* `WavFileTest.WriteThenRead` (`wave_gtest.cpp:192`) ends with `// optional: compare contents`, so it could not catch this. The FSDD window tests check counts and z-score, not audio content.
- *History.* The repo log for `Wav.cpp` shows reorganization commits only, so this is probably old and earlier speech numbers carry it, including whatever speech figures went into rejected submission 71. I did not check that submission.
- *Not affected.* EEG, which uses its own EDF reader (`Meeting01Eeg.cpp`). Other `Wav` users are `src/core/wave/demo` and tests.

**Smallest fix.** Replace the seek at `Wav.cpp:373` with `ifs.seekg(sizeof(header))`, and throw unless the `data` chunk tag sits at the expected offset. A file with extra chunks must fail loudly. Add a known-answer test: write known samples, read them, compare values. Bump `RESULTS_FORMAT` to 10 so that a resume cannot mix old and new rows. On AudioMNIST, confirm that the `sox` output has a 44-byte header; I could not, since the data is not on this machine.

### D2. Window caps truncate instead of sampling (V002). CONFIRMED; size depends on dataset

**Code.** `stratified_window_cap` (`Meeting01Dataset.cpp:126`) walks a `std::map` keyed by recording id and breaks at the cap. With fewer slots than recordings, the first pass never reaches the later ones.

**Measured on real metadata (`v002.py`).**

| Split, fold 0 | FSDD | EEGMMIDB |
|---|---|---|
| Train | 1000 of 2000 recordings, 1 window each: **digits 0 to 4 only**, 4 speakers equal | 1000 of 1008 recordings, 72 subjects: subject S109 loses 8; harmless |
| Validation | 150 of 500: **digits 0 to 2 only**, 1 speaker | 150 of 252: **11 of 18 subjects** |
| Test | 500 recordings, 3 windows each, all digits | 266 recordings, 5 to 6 windows each, all 19 subjects |

The reviewer's "≈1022" EEG pool should read 1008 (72 × 14).

**AudioMNIST, port of the code on the documented layout `NN/d_NN_t.wav` (`audiomnist_sim.py`).** Not run on real files.

| Split | Pool | Kept |
|---|---|---|
| Train | 40 speakers | **Speakers 01 and 02**, in every fold where they train |
| Validation | 10 speakers | **First speaker of the block, digits 0 to 2** |
| Test | 10 speakers | **First 3 speakers of the block** |

So "leave-one-group-out on 60 speakers" would train on 2 and test on 3. That is fatal.

**Siena.** Not fatal by this route: few recordings at 40 windows each, so the first pass reaches all of them. D3 is its problem.

**Effect on FSDD: small, and within window-draw noise.** Clean windows, `clean_probe.py`, same test set:

| Selection | PCA k=16 | PCA k=64 | ANN-AE | SNN T=4 | Gap |
|---|---|---|---|---|---|
| Truncated (as saved) | 0.707 | 0.2425 | 0.715 (2 seeds) | 0.779 (2 seeds) | 0.064 |
| Balanced draw 1 | 0.695 | 0.2409 | 0.738 | 0.790 | 0.052 |
| Balanced draw 2 | 0.648 | 0.2412 | 0.703 | 0.762 | 0.059 |

Two things follow. The ANN-AE moved 0.035 between two balanced draws and PCA k=16 moved 0.047, so *which 1000 windows* is itself a large source of variance the paper does not report. And digits 5 to 9 err more than 0 to 4 in every arm, balanced or not (PCA k=16, truncated 0.74 vs 0.67, balanced 0.69 to 0.73 vs 0.61 to 0.66): that is digit difficulty, not an unseen-digit penalty. Reviewer's FATAL is right as a defect, and for AudioMNIST. It overstates the FSDD numbers.

**Smallest fix.** Inside `stratified_window_cap`, shuffle the recording order (seeded) before the round-robin, and stratify by group, and by digit where a label exists. Add a test: for a pool much larger than the cap, the kept recordings cover every group, and every digit where available. Print composition per fold into the manifest. The window draw should be a seed-varied factor; report its spread.

### D3. Per-recording cap keeps the first windows (V013, V017). CONFIRMED by code

`Meeting01Dataset.cpp:225`: `if (source_window_index >= max_windows_per_recording) continue;`. EEGMMIDB keeps the first 64 s of each run, Siena the first 20 s of each file, AudioMNIST windows 0 and 1. For AudioMNIST, even after D1, the wiki records window 0 as a near-silent lead-in that z-scoring amplifies. For Siena, this picks whatever sits at the start of each file, not a sample of the recording.

**Smallest fix.** Choose the kept indices by seeded sampling across the recording. For AudioMNIST, sample inside the voiced region or drop the cap. Then state it in the paper.

### D4. 128 Hz EDF files processed as 160 Hz (V013)

`EegWindowDataset` passes the profile's `sampling_rate` to `bandpass_notch` (`Meeting01Eeg.cpp:197`) for every file. 36 of 1526 files, subjects S088, S092 and S100, are at 128 Hz, so for those files the 40 Hz edge lands at 32 Hz, the 60 Hz notch at 48 Hz, and a window lasts 2.0 s, not 1.6 s. In fold 0 that is 42 of 1000 training windows and none of test or validation. Those subjects fall in test blocks 4 and 5 (`rank*6/109`), so headline folds 4 and 5 see them directly.

**Smallest fix.** Exclude the three subjects and state it, which is cheaper than per-file rates and avoids mixed window durations. The profile comment should change too.

### D5. PCA has no energy (V008, m16)

`02_` writes the energy verdict for SNN-AE and ANN-AE only, so the "Energy first" rule is undefined for the reference most likely to beat them. My ledger prices PCA k=16 at 158 nJ and k=64 at 903 nJ against ANN [59,16] at 854 nJ and [65,64] at 1079 nJ. **Fix:** add a PCA row (encode plus decode) to `02_` and the table.

### D6. Reporting gaps in the analysis (V010, M13, V005)

`04_` computes speaker-level results into JSON only (`speaker_level`, line 241). There is no sensitivity run without fold 1. **Fix:** print per-group effects in the table, make them primary, add the fold-1-out sensitivity. Note that on FSDD this leaves 4 speakers, with minimum exact p = 0.125.

### D7. Re-score keeps the search seed (M9)

Real, cheap, low impact. See M9 below for why the reviewer's numbers are wrong. Selection quality is unchanged; the reported validation score is about twice as biased as with fresh seeds; test numbers are unaffected. **Fix:** in `Meeting01GaSearch.hpp`, start `mse_sum` at 0 and loop over fresh seeds. No result is invalidated, so this one should not delay anything.

### D8. Stale inputs

- `snn_gap_ablation_torch.py:59` loads the saved `.npy` windows, so the design study inherits D1 and D2. It also keeps last weights and uses plain patience, unlike the harness.
- T1's `WINNERS` come from job 5315335 (format 7).
- T1's `share_windows_10pct_below_mean` on FSDD counts the header.

**Fix:** regenerate Table `tab:design` after D1 to D4, with logged, archived output, and rerun T1 on current winners.

### D9. Small

- `LifBPTT::compute_backward_params` does not clamp V_th, while the paper says it does (m20). Fix the text.
- `LifBPTT.hpp:~411`, comment "Exact local derivative": false. The branch also treats the two terms inconsistently, since `dvpost_dVth` keeps the surrogate-through-reset term and `dvpost_dvpre` drops it. Whether that matters is untested; I found no evidence it does. Fix the comment; a two-arm check is optional.
- Profile `_preprocessing` comment ("every speaker/recording stays represented") is false until D2. `fig_pareto.tex` header still says arithmetic energy.

### Not defects, but design limits that must be stated

- Per-window z-score plus min-max removes input sparsity (V006).
- The latent time mean discards spike timing (V012).
- Budget and window draw move the gap (V014, D2).

---

### Job 5316016: usability

I have no cluster access. This assumes the job runs format 9 at HEAD and writes per-window CSVs as T1 did; check that when collecting it.

| Dataset | Family | Reported MSE | MSE on windows ≥ 1 | Inference time, E | Training wall time |
|---|---|---|---|---|---|
| FSDD | SNN-AE | Not usable | Decision-grade only | E indicative (winner may change) | Indicative |
| FSDD | ANN-AE | Not usable | Decision-grade only | Usable for that architecture | Indicative |
| FSDD | LSTM-AE, GRU-AE | Not usable | Decision-grade only | Usable for that architecture | Indicative |
| FSDD | Transformer-AE | Not usable | Decision-grade only; "armed" flags not meaningful | Usable for that architecture | Indicative |
| EEGMMIDB | SNN-AE, ANN-AE, LSTM-AE, GRU-AE, Transformer-AE | **Usable** with disclosure | Not needed | Usable | Usable |

"Indicative": stopping time depends on D1's skewed arming. "Decision-grade": fit to judge whether the removal of sequence families still stands, not to print. On EEGMMIDB, the disclosures are D2 (validation covers 11 of 18 subjects), D3 (first 64 s) and D4 (4.2% of training windows at the wrong rate). None invalidates it.

**Recommendation.** Let the job finish. Use EEGMMIDB as is. For FSDD, filter to windows ≥ 1 for the go/no-go, and rerun FSDD after D1 to D4 before printing a row. Do not start the main grid before D1 to D4.

---

## Part B. Objection by objection

### Reviewer 1

**V001. CONCEDE.** Evidence in D1. The reviewer's shift, PCA, mean, window share and window-0 errors reproduce; mean pairwise window-0 correlation in test is about 0.92 on the first 50, against his 0.89. His gap of 0.082 does not hold up: 0.064 on 2 seeds.
*Status:* C114, C133, C273 → WRONG as statements about speech. C006, C076, C183, C272, C289 → CONDITIONAL on D1.

**V002. CONCEDE, with the size trimmed.** Evidence in D2. Reviewer's 0.707 to 0.703 is one draw; two draws give 0.695 and 0.648, so PCA is sensitive to window draw and truncation can't be separated from it here.
*Status:* C182 SUPPORTED → WRONG as a description of coverage (the phrase "round-robin" is literally true, and misleading). C011, C048, C184 → CONDITIONAL on D2.

**V003. CONCEDE, and the reviewers' conflict resolves.** Reproduced from `grid.json`: all rows of his table, with excess and shares as stated. New runs add what neither reviewer measured: how saturated the sigmoid actually is.

| Same init, share of hidden values at ≤ 0.01 or ≥ 0.99 | EEG excess | EEG share | Clean FSDD excess | Clean FSDD share |
|---|---|---|---|---|
| Sigmoid | +0.008 | **0.3%** | +0.034 | **3.9%** |
| LIF T=16 | +0.108 | 54% | not run | |
| LIF T=4 | +0.153 | 76 to 77% | +0.064 | 90% |
| Clamp | +0.195 | 89% | +0.084 | 86% |
| Clamp, identity gradient | +0.248 | 98 to 99% | not run on clean | |

(EEG excess against leaky ReLU, 2 seeds from the grid, 1 seed for the new sigmoid share. Clean FSDD: 1 seed for sigmoid and clamp, 2 for LIF.)

**Resolution of F2 vs V003.** F2 argues a sigmoid is bounded and saturating, yet stays near the control, so saturation is contradicted. That reads a property of the function, not of the trained network. A trained sigmoid here is *not saturated*: 99.7% of EEG values sit away from its ends. So it says nothing against saturation as an operating state. Equally, the share of pinned units tracks the excess on EEG in monotone order, which fits V003, but pinned-at-0-or-1 and near-binary are the same event. Saturation and quantization are not separated. F2's own comparison also crosses Kaiming and Xavier cells. Both reviewers are right that the sentence is not earned; F2's "contradicted" is too strong and so is any reading of V003 that acquits the spike.

**New, against my side.** On clean FSDD the clamp (0.795) is *worse* than the SNN (0.779), as on EEG. The "clamp matches the SNN on FSDD" result the reviewer credited was partly the header. Single seed.

*Status:* C026, C292, C322 stay UNSUPPORTED. C294 → PARTIAL (direct input, replica only). C235, C236 stay WRONG. Pressure point 3(e) on dead gradients withdrawn: identity gradient makes it worse on both datasets, in the reviewer's table.

**V004. CONCEDE.** Same substance as F3. No new evidence needed. *Status:* C007, C045, C046, C105, C002 stay WRONG. The clamp-gradient effect of 0.043 to 0.053 reproduces from the grid.

**V005. CONCEDE, and P19 is too cheap.** Resolution of the reviewers' conflict, from `assign_speaker_fold` (`Meeting01Dataset.cpp:20`) and the T1 manifest:
- Reviewer 2 is right on one fact: the test block of fold 0 is in no headline test.
- He read no split code, and the conclusion does not follow. Block 1 was fold 0's validation block *and* is headline fold 1's test block. That is direct contamination, as C194 says. FSDD: jackson. EEG: subjects 19 to 36.
- Blocks 2 to 5 trained the dev models, so all speakers touched the design, though this path is much weaker than block 1.
- Added, unreported by either: block 0 is headline fold 5's validation block (`(5+1) mod 6`), so a block the designer looked at also selects a headline model.

*Status:* C194 stays WRONG. C021, C057, C195, C317 → PARTIAL. Undisclosed dev choices (stop skill 0.10, 1000 windows, T range, surrogate sharpness) must be listed.

**V006. CONCEDE, with corrected numbers (`v006_m5.py`).** [59,16]:
- Per synapse: 27.1 pJ for an analog MAC with input read (409.5 nJ over 15 104 synapses) and 43.2 pJ for a synaptic event, not 41 and 57. His ratio is close, his absolutes are not.
- Break-even input spikes per sample per window: 0.60 at T=2, 0.585 at T=4, 0.55 at T=8, 0.47 at T=16, 0.36 at T=28. The reviewer's 0.5 to 0.6 holds only at low T.
- Latency is exactly 1 by construction (`Meeting01Encoding.cpp:82-93`): always above break-even. First layer 670 to 830 nJ against 409.5.
- Poisson p: 0.47 to 0.51 on FSDD, 0.42 to 0.44 on EEG, not 0.50 to 0.55. At T=2 that is 0.85 to 1.0: still above break-even.
- Direct: first layer never below the ANN's. The maximum saving is in Lin(h→d), and my best case is T=2 at 10% rate: total ratio 0.997.

*Not verified:* the claim that only the `recurrent` transform can reach the winning region, and any depth above one hidden layer.
*Status:* C300 SUPPORTED → PARTIAL. Introduction's "does no work" (line 76) → PARTIAL under this normalization.

**V007. CONCEDE.** 430 of 854 nJ reproduced. 1.53× total implies 2.07× encoder-only. *Status:* C031 PARTIAL stays; C158 arithmetic stays SUPPORTED; add encoder-only column.

**V008. CONCEDE.** Evidence in D5. Even on 266 recordings the naive SE is 0.0019 (ICC run below), so with 2500 recordings the first branch is effectively unreachable. *Status:* C013, C204, C205 → PARTIAL; C206 stays WRONG.

**V009. CONCEDE.** EEG: ANN-AE 0.289 against PCA 0.1825, computed from saved windows; the mean 1.0005. FSDD clean: 0.715 against 0.707. *Status:* C233 → UNSUPPORTED. C110, C111, C232 → PARTIAL.

**V010. CONCEDE.** 3 windows per FSDD test recording, 5 to 6 EEG, confirmed. *Status:* C012, C198 to C203, C302, C303 unchanged; C200 needs the group column.

**V011. CONCEDE.** Decomposition of all 12 T1 FSDD final fits (`v011_icc.py`):
- 11 fits: window 0 at 0.112 to 0.272, other windows 0.990 to 1.009.
- 1 fit (c1 B, seed 43): 0.116 and **0.9305**.

The clean mean predictor is 1.0008. So on speech, 10 of 12 fits are at the mean predictor; c1 C seed 43 at 0.990 is marginal. This makes the plateau on FSDD *stronger*, not weaker, as a finding about real speech, and it voids "0.935 against 0.996" as evidence of learning.
*Status:* C276 FSDD half → WRONG; EEG half stands. C272, C273 arithmetic stands, meaning falls. C277 stands. Note the denominators (m15).

**V012. CONCEDE.** True by construction: the latent is a time mean. *Status:* C242, C291, C297 → PARTIAL; add a limitation.

**V013. CONCEDE.** D3, D4. *Status:* C077, C174, C185 stay as in the dossier, C102 → PARTIAL.

**V014. PARTIAL.** The window table (wiki, lines 1364 to 1372) is real, but it is format 4: both rows predate the current design, so it does not measure the adopted model's budget dependence. The first windows run even goes down (0.174 to 0.161 from 200 to 2000 windows) before rising. EEG best epochs at 19 to 30 of 30 are confirmed and the cap is already Limitation (vii). What stands: the gap depends on budget and the paper should say so; D2 adds window draw. *Status:* C046 → PARTIAL; C313 stands.

**V015. CONCEDE.** Figure plan in Part E.

**V016. CONCEDE.** All already WRONG in the dossier.

**V017. PARTIAL.** Untestable here: AudioMNIST is not on this machine. For it: 1 unique spike pattern in 2700 windows (wiki, Meeting01.md) is what a shared header would produce. Against certainty: the wiki also reports raw lead-in values of ±2 to ±12, which reading the WAV file with another tool would show even if the loader was reading a header, so it doesn't decide the case. Both may be true, and a header would dominate a quiet window's z-score. I could not find `check_degenerate_reconstruction` under `src/experiments/meeting01/lib`, so its threshold is unchecked. *Severity:* MAJOR if AudioMNIST stays. *Status:* C184 → CONDITIONAL on D1 to D3.

**V018. CONCEDE.** The code comment agrees: contiguous groups. *Status:* C010, C047 → PARTIAL.

**V019. CONCEDE.** *Status:* C116, C118, C127 unchanged.

**V020. CONCEDE.** Replica: V_th 0.6 to 0.8 and 0.96 to 2.2 to 2.6, from `grid.json`. Replica parametrization differs. Print trained values from the harness. The paper's clip argument uses initial values: a further reason to drop it.

**V021. CONCEDE.** Text already in P06, P07.

**V022. CONCEDE.** Text already in P02, P19, P24, P27, P29.

### Reviewer 2

**F1. PARTIAL.** Placeholders are real and the paper can't be submitted without the grid. But the premise is off: "Spiking autoencoders cost more than they save" is not this paper's title, and the abstract does not claim a measured result; it has a TODO in that slot. The mixture is a draft state, not an assertion. The real fault is narrower: "shows" for the design study. *Status:* no claim change; P01, Q1.

**F2. PARTIAL.** The transfer-function derivation is right, and it is the same result as V003's. Working k=2 at I=V_th by hand confirms it: the paper's "fires at every step for I ≥ V_th" is wrong at equality given strict `>`. "Contradicted by the paper's own design table" does not hold, for two reasons: its comparators cross cells (per-step at T=16 Kaiming against a sigmoid), which is the same flaw as C235 and C236; and a trained sigmoid is not saturated (table under V003). "Hard decision boundary" is also not established: clamp has a hard edge, not a decision boundary, and its output is continuous inside the range. *Status:* C241 stays WRONG; C026 UNSUPPORTED, not WRONG.

**F3. CONCEDE.** Same as V004. His decomposition is the right statement of what the code does, and adds time-mean as a term. *Status:* as V004. One addition: the control also inherits widths selected for the SNN, which he does not list.

**M4. PARTIAL.** The paper's Eq. 2 is not what the code computes, and the comment is false: conceded (C092, D9). Three corrections:
1. His numeric column for the paper is wrong. With his inputs the paper gives 1.5 at both points, not 0.5 (v=1.2) and 0.5 (v=0.9). Code 0.5 and 1.5, "exact" −0.1 and 1.05 are right.
2. There is no "exact" derivative: dS/dv is a delta function. His expression is the surrogate carried through the reset, one choice among others. The code detaches the reset, the common practice (snnTorch's default), and that is defensible.
3. A finite-difference check cannot work: the loss is piecewise constant in v_pre at a threshold. Only the forward-pass-with-surrogate identity can be checked, not the gradient of the hard model.

*Status:* C092 stays WRONG; P09 stands; the paper should say "the reset is detached", not claim exactness.

**M5. PARTIAL.** Direction right; magnitude trivial; "error" overstated. SNN/ANN ratio under memory scaling (`v006_m5.py`):

| Memory scale | direct T=4 | direct T=28 | Poisson T=4 | Poisson T=28 | Latency T=28 |
|---|---|---|---|---|---|
| 1.0 | 1.054 | 1.478 | 2.196 | 12.35 | 1.912 |
| 0.75 | 1.050 | 1.458 | 2.117 | 11.80 | 1.860 |
| 0.5 | 1.044 | 1.424 | 1.983 | 10.86 | 1.770 |

Halving memory lowers the ratio by 1 to 12% and crosses 1 nowhere. Note halving is an assumption: Horowitz gives no 32-bit figure, and bit-line energy scales with width while decoding does not, so 0.5 is a floor, not "the 32-bit price". The disclosure is conceded (P07), and adding a 0.5 sensitivity is cheap. *Status:* C148 → PARTIAL. C014 stands.

**M6. PARTIAL.** Line 277's sentence is false as a blanket, conceded (C151 WRONG, P08). "Not disclosed" is wrong: line 287 says "This choice favours the SNN-AE", and the abstract announces the repeat with the first layer recomputed at every step. Magnitude reproduced: T=28, 1.48× once against 14.37× as executed. The objection's demand (report both) is already the design.

**M7. PARTIAL.** Line 286 states the write and the read at every step, so E is reproducible from the prose. The table lacks the row. Add it (extends P15).

**M8. PARTIAL.** Encoder and leaky ReLU are priced in prose (lines 280 to 283). The Mean note is wrong (C173, P16). "Undercharges to the SNN" is wrong: the code charges them, and only the table under-describes. So the "conservative" point does not follow from this.

**M9. PARTIAL. His simulation has a bug.** In `checks.py`, line 118 draws the fresh scores as noise without μ, and so does line 124. That produces his −1.118 and ≈0.00 (`m9_m13.py` reproduces both from his formula). Corrected (`m9_m13.py`, `m9_front.py`), 48 candidates:

| | E[true quality of selected] | Selected score minus truth |
|---|---|---|
| Single seed | −1.58 | −1.58 |
| 1 search + 2 fresh, all re-scored | −1.935 | −0.643 |
| 3 fresh, all re-scored | −1.936 | −0.643 |
| 1 + 2, front only (as coded; front ≈ 4.5) | −1.75 | **−0.585** |
| 3 fresh, front only | −1.75 | **−0.297** |

Noise sd equal to candidate sd. So keeping the search seed does not change which model wins; "removes about 29% of the bias" is wrong; and "unfair" only applies to the reported validation score, which is about twice as biased. Test numbers are unaffected. The conceded part: "3 seeds" must be described correctly (D7). *Status:* C118, C119-equivalent stay PARTIAL.

**M10. PARTIAL.** His counterexample has the sign reversed: lower is better, so a selected family at −1.58 sd *wins*. In true-quality terms, more search does favour it. That is the opposite of what he states. "Not a theorem" still stands: nothing proves the inequality, and the control's widths were selected for the SNN, which may hurt it. So "conservative" goes (P05). *Status:* C111 → PARTIAL with P05.

**M12. CONCEDE.** Text false: FSDD T=4, Kaiming 0.724, Xavier 0.729. The difference is noise-sized, but "better or equal for each arm on both datasets" can't stand. At T=16 Kaiming EEG the SNN *beats* the control, so "looked equal" depends on T. *Status:* C232 → WRONG.

**M13. CONCEDE, and the data make it worse than he argued.** Two runs:
- Simulation, 5 speakers × 500 recordings, no true effect (`m9_m13.py`): recording-level Wilcoxon false-positive rate 5% at ICC 0, **40% at ICC 0.01, 69% at 0.05**; speaker-level t stays at 5 to 6%.
- Empirical ICC on real paired errors, LIF T=4 vs ANN replica, EEG fold 0 (`icc_probe.py`): **0.33**, design effect 5.3, SE 2.3 times the naive one.

"Invalid" is too strong in one sense: conditional on these speakers it is a valid test of these speakers' recordings, as Limitation (i) says. But the paper uses it to decide, so it should not be primary. This effect would survive anyway: 19 of 19 subjects positive, d=0.154, cluster SE 0.0044. The risk is at small gaps, which is where "Energy first" lives. A prior proxy pair of T1 trainings gave ICC ≈ 0 because both sat on a plateau; it is uninformative. *Status:* C198 → PARTIAL, C302 SUPPORTED as disclosure.

**M14. CONCEDE.** *Status:* C206 WRONG. Use a mean-based estimand, P21 with margin.

**M17. CONCEDE.** Same as C120, P11.

**M18. CONCEDE.** Regeneration with D8 should cross loss × decoder and init × activation.

**M19. PARTIAL.** The quoted text is not in the paper, and the table does separate them sequentially at T=16: .780 → .734 (loss) → .729 (decoder) on FSDD, .592 → .454 → .457 on EEG. So his "not separable" fails, under the caveat that latent mean and output mean differ. But reading it that way exposes something worse that neither of us caught: **the decoder changed error by −0.005 and +0.003.** "A spiking decoder added error", and the abstract's "two avoidable choices enlarged the gap", are not supported for the decoder. Its effect is cost. *Status:* new: the decoder half of design finding (i) → UNSUPPORTED. Q7.

**m15. CONCEDE.** Add V011's correction. *Status:* C276 per V011.

**m16. CONCEDE.** D5.

**m20. CONCEDE.** Same as C088.

**m22. CONCEDE.** Say "48 candidates", or report distinct trainings.

**m23. REBUT.** The profile exists: `software/nn/src/experiments/meeting01/profiles/meeting01-loso.json`. `voltage_threshold` 0.5 to 1.5, alpha 0.8 to 0.99, time_steps 2 to 32, population 8, generations 5, winner_seeds 3. The macros match. His "Could not verify 1" is closed.

**m24. CONCEDE.** Trivial: "at or below".

**m25. REBUT.** The paper says only that Lemaire et al. do not count the threshold test (line 279 to 280), which is true. It attributes the add to nobody. Hoefflin & Wassner appear only in a code header. Nothing needs citing.

**m26. REBUT.** Not a paper claim, and the direction is reversed: removing the restore (`Meeting01Experiment.cpp:535`) means every run recomputes its rows, so nothing stale can reach a table.

**m27. CONCEDE.** One clause.

**m28. CONCEDE.** "After seeing the development folds", in P21.

### Reviewer claims resolved

| Conflict | Resolution |
|---|---|
| Split: "correct" vs contaminated | V005: fold 0's test block is excluded; validation block 1 is contaminated. Both true of different blocks. |
| m23 profile | Exists; macros match. REBUT. |
| Saturation | V003 and F2: sigmoid units are not saturated; neither "contradicted" nor "supported"; not separated from quantization. |

---

## Part C. Changes to the paper (old → new)

P01 to P33 of the dossier stand, except that **Q1, Q3, Q4, Q9, Q17 replace P01, P22, P23, P21, P24**. Bracketed numbers wait for logged reruns after D1 to D4. Nothing below has been applied.

**Q1 (abstract, lines 59 to 64).**
OLD: "shows that two avoidable choices -- scoring every simulation step and a spiking decoder -- enlarged the gap, that the initializer decided the EEG comparison (with fan-in scaling in both arms the control was handicapped and the gap vanished), and that what remains is reproduced by a non-spiking network with a hard-clipped activation: the cost is the saturation of the neuron's transfer function, not the spike."
NEW: "shows that scoring every simulation step enlarged the gap and a spiking decoder added cost without improving error, and that the initializer decided the EEG comparison. On a direct-coded, one-hidden-layer design the remaining gap tracks how close the hidden code is to binary; this does not separate saturation from quantization."

**Q2 (Setup, lines 331 to 333).**
OLD: "long recordings are capped to a number of windows, and each fold keeps at most ... test windows, sampled round-robin across recordings."
NEW: "long recordings are capped by [sampling that number of windows across the recording], and each fold keeps at most ... test windows, sampled at random within each group and stratified by digit where digits are labelled. Composition per fold is in the released manifests. EEG is band-passed at 0.5--40 Hz and notched at the mains frequency; EEGMMIDB subjects S088, S092 and S100, recorded at 128 Hz, are excluded."

**Q3 (design (iii), replaces P22).**
OLD: from "\emph{(iii) What remains is saturation.}" to "a clipped transfer function."
NEW: "\emph{(iii) What remains is not yet explained.} With the initializer shared and only the encoder's activation changed, a sigmoid, whose trained values rarely reach the ends of its range ([0.3--4]\%), stays near the control ([+0.008, +0.034]), while a hard clamp and an LIF, whose values sit at 0 or 1 in [54--90]\% of cases, lose [0.06--0.20], with the excess rising with that share. A clamp is worse than the SNN on both datasets, so a clip does not reproduce the gap. An LIF under constant current is a staircase with levels 1, 1/2, 1/3, \dots, and no level between 1/2 and 1 at any $T$. These runs use direct input, so they do not cover the Poisson winners."

**Q4 (Discussion, replaces P23).**
OLD: "The spike's cost in this autoencoder is not a property of discrete events as such: ..."
NEW: "The design study does not separate the spike's discreteness from the saturation it brings: a hidden code near 0 and 1 is both. Remedies that move the code off those values are candidates to test."

**Q5 (Discussion, energy).**
OLD: "binary inputs save the first layer's multiplies while every step pays for the potentials, so whether the SNN-AE ends up cheaper than its control depends on the encoding, $T$ and spike rate the search selects, not on spiking alone."
NEW: "On these normalized windows a spike-fed first layer costs more than an analog one: a synaptic event costs [1.6] times a multiply-accumulate once memory is priced, so it wins only below [0.36--0.60] input spikes per sample per window, while latency coding gives 1 by construction and Poisson coding about [0.45]$T$, because per-window normalization makes silence as dense as signal. For one hidden layer, direct coding saves at most [0.3]\% of $E$."

**Q6 (Introduction, line 76).** After "does no work" add: ", a saving that requires sparse input". And add Limitation (ix): "\emph{Rate read-out.} The latent is a time mean, so the timing of spikes carries nothing here; conclusions concern rate-read spiking."

**Q7 (design (i)).**
OLD: "A spiking decoder added error on top of the encoder's; averaging the latent and decoding once with leaky ReLUs is as good and runs the decoder once instead of $T$ times."
NEW: "Replacing the spiking decoder by leaky ReLUs left error unchanged ([FSDD $-0.005$; EEG $+0.003$]) and runs the decoder once instead of $T$ times."

**Q8 (design (ii)).**
OLD: "Xavier is the better or equal choice for each arm on both datasets, so the matched comparison is also a best-against-best one."
NEW: "Xavier is better for both arms on EEG and within noise on FSDD; neither initializer was designed for LIF neurons, so this is the better of two."
Also OLD: "looked equal to it (0.484)" NEW: "matched it at $T=4$ (0.484) and beat it at $T=16$ (0.457)".

**Q9 (Energy first, replaces P21).**
OLD: paragraph "\textbf{Energy first.} ..." NEW: "\textbf{Reading the two axes.} We fixed this after seeing the development folds and before the main run. For SNN-AE against ANN-AE, where the group-level interval for the mean difference lies inside $\pm[\Delta]$, the model with the lower $E$ is recommended; otherwise we report what the lower-energy model gives up. On FSDD, with [4--5] speakers, no test can reach $0.05$, so only the interval is read. PCA is priced by the same model."

**Q10 (statistics; Limitation (i)).** OLD: "Group-level and seed-level tests are reported as robustness only" NEW: "Per-group effects are primary; the recording-level test is descriptive, since an intra-class correlation of [0.33] between subjects inflates its precision about five-fold in variance."

**Q11 (Setup, dev folds, strengthens P19).** Add: "All speakers of FSDD entered the development fold as test, validation or training, and the development validation block is the test block of fold 1, so FSDD and EEGMMIDB results are design-informed. Fold 5 validates on the development test block. A sensitivity analysis drops fold 1. AudioMNIST and Siena were not used in design." Add the list: stopping skill, 1000 windows, $T$ range, surrogate sharpness.

**Q12 (naming, lines 110, 628).** OLD: "nested leave-one-group-out cross-validation" NEW: "nested group-disjoint six-fold cross-validation (leave-one-speaker-out on FSDD)".

**Q13 (Search).** OLD: "re-scored over \cfgGaWinnerSeeds{} seeds" NEW, after D7: "re-scored over \cfgGaWinnerSeeds{} fresh seeds". If D7 is skipped: "over the search seed and two fresh seeds".

**Q14 (T1 verdict).** After D8, or if not rerun, add for FSDD: "On FSDD, [10 of 12] final fits are at the mean predictor once the first window of each recording is set aside." Add the PCA reference.

**Q15 (Energy).** After P07: "Halving every memory price lowers the SNN/ANN ratio by [1--12]\% and does not change its side of 1." And m24: OLD "Memories below 8\,KB" NEW "Memories at or below 8\,KB".

**Q16 (Table cost).** Add a row for the $T\,n_{\mathrm{out}}$ read-backs of a stored first layer (M7).

**Q17 (conclusion, replaces P24).** OLD: "-- and located the remaining cost in the saturation of the LIF transfer function." NEW: "and left the remaining cost tied to how binary the hidden code is, without separating saturation from quantization."

---

## Part D. Claims to remove

| Claim | Why |
|---|---|
| "the cost is the saturation ... not the spike" (C026, C292, C322) | Not separated; Q1, Q4, Q17 |
| "located the remaining cost" | Same |
| The clip argument "fires at every step for I ≥ V_th" (C241) | Wrong at equality; a staircase; Q3 |
| "only the encoder's hidden neuron differs ... the cost of spiking" (C007, C045, C046, C105) | P03, P04 |
| "works against the conclusion it could support" (C151) | P08 |
| "a gap in the ANN-AE's favour is conservative" (C111) | P05, M10 |
| "best-against-best" (C233) | Q8 |
| "A spiking decoder added error" | M19, Q7 |
| "Xavier is the better or equal choice" | M12, Q8 |
| "every choice made on validation groups" (C316) | P19 |
| "the information available is the same" (C311) | P29 |
| "fixed in advance" (C206 context) | m28, Q9 |
| "usually compared ..." (C003) | P02, or cite |
| "sampled round-robin across recordings" (C182) | D2, Q2 |
| AudioMNIST and Siena results, if D2 and D3 aren't fixed and verified on real data | Better to drop them than print them |

## Part E. Figures accepted

About 48 cm of column remain (1 page). Accepted, about 21.5 cm net, leaving room for the text growth above:

| Figure | Space | Note |
|---|---|---|
| m8, corrected Fig. 1 | 0 net | Replaces Fig. 1; P04 becomes unnecessary |
| F1 LIF ladder, with share of pinned units | 4.5 cm | Left now; right from the rerun of Q3 |
| F2 Fold map and cap coverage | 4.5 cm | After D2, shows what is kept |
| F3 Energy blocks and break-even | 4.5 cm | Use my corrected curve (0.36 to 0.60) |
| F5 Design dots and T1 strip | 3 cm net | Replaces Table tab:design; show dispersion |
| F7 Per-group forest plot | 5 cm | Primary display for Q10 |

Declined for now: F6 (a sweep rerun, and a sentence does it, unless the run shows otherwise), F8 (one sentence, Q6), F9 (first to go, per the reviewer). Fig. 2 stays, with winners as points, a PCA point (D5) and the ANN width sweep only if it can be run.

---

## Not checked

- AudioMNIST and Siena: no data here. D2 and V017 for them come from code and a port. Siena's recording count comes from the profile cap logic, not the files.
- Effect of D1 on a Poisson or latency winner, and on sequence families in the harness: only T1 Transformers were decomposed.
- Replica numbers on clean FSDD: 1 to 2 seeds. The clamp-vs-SNN reversal is one seed.
- ICC: one model pair, one fold, one dataset, replica.
- Job 5316016 itself, and the claim that only the `recurrent` transform can win on energy.
- Submission 71's speech figures.

Scripts: `dossier/response_scripts/` (`v001.py`, `v002.py`, `audiomnist_sim.py`, `clean_probe.py` and `clean_probe.json`, `v011_icc.py`, `icc_probe.py`, `m9_m13.py`, `m9_front.py`, `v006_m5.py`, `eeg_misc.py`).
