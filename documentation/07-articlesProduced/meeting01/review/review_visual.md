# Reviewer 1 report: "What Does Spiking Cost an Autoencoder?" (format 9 draft)

Reviewer 1 (visual). Independent of the other reviewer; `review_math.md` was not opened.
Date: 2026-11-04. Mockups: `dossier/visual_mockups/` (m1 to m9, each stamped MOCKUP or EVIDENCE). Probe code and raw output: `dossier/visual_mockups/probe_src/`.

**Bottom line.** MAJOR REVISION now, and REJECT if submitted before V001 and V002 are fixed. Two silent data defects the Author missed sit under every speech number (V001, V002). The headline causal sentence fails as written (V003). The paper has two figures, one a placeholder, and neither shows any mechanism it argues.

**How I checked.** I read the code and the wiki, recomputed from the T1 files, and re-implemented the energy table. I also trained a small numpy replica on the harness's own saved windows. That replica is a reviewer probe, not harness output. It has plain best-validation restore, β trained directly rather than through R and C, and 2 seeds, except where stated. It reproduces the paper's numbers closely, so I use it as evidence of direction:

| Xavier, test MSE | Paper, FSDD | Probe, FSDD | Paper, EEG | Probe, EEG |
|---|---|---|---|---|
| ANN-AE | 0.654 | 0.667 | 0.293 | 0.289 |
| SNN latent mean, T=4 | 0.729 | 0.727 | 0.449 | 0.442 |
| SNN latent mean, T=16 | not run | not run | 0.396 | 0.397 |

---

## 1. Objections

### FATAL

**V001 · FATAL · new; touches C006, C076, C114, C133, C183, C272, C273, C289, all FSDD rows of Table III, the 0.649 PCA figure · The WAV header is fed to the networks as audio.**
- *Objection.* `Wav::read_wave_headers` ends with `ifs.seekg(0, std::ios::beg)` (`src/core/wave/Wav.cpp:373`), and `read16BitMono` then reads from byte 0. The 44-byte RIFF header becomes the first 22 "samples" of every recording, and the last 22 real samples are dropped. Every FSDD window is 22 samples late, and window 0 of every recording is dominated by a near-constant pattern. The paper describes these as speech windows.
- *Evidence.*
  - Saved window (rid 50, index 7) correlates 0.99999 with the WAV at offset 1770, not 1792. Rid 51, 52 and 58 show the same shift of exactly 22.
  - Window 0 of 25 test recordings, z-scored, shows the bytes of "RIFF...WAVEfmt" at values up to 7.2 (m7). Mean pairwise correlation among window-0s is 0.98 in train and 0.89 in test. They are 8.2% of train and 7.8% of test windows.
  - Rebuilding the same (recording, index) windows without the header moves PCA from 0.649 (exactly the Author's number, reproduced from saved windows) to 0.707, and the mean predictor from 0.9955 to 1.0008.
  - PCA errs 0.095 on window 0 and 0.696 elsewhere.
  - Header-free training (1 seed) gives ANN 0.710, SNN T=4 0.792, gap 0.082, against 0.060 with the header. Without the header the ANN-AE is about PCA (0.710 vs 0.707).
- *Satisfied by.* Fix the reader, add a known-answer test against a generated WAV, regenerate all speech numbers including the design study, and state the correction. The EEG path is not affected: it uses its own EDF reader.
- *Verdict.* DISAGREE with every status that treats the FSDD figures as describing speech (for example C114 "≈1.0", C133, C273). The Author missed this.

**V002 · FATAL · C182 (SUPPORTED), C011, C048, C184, profile comment "every speaker/recording stays represented" · The window caps truncate by file order instead of sampling.**
- *Objection.* `stratified_window_cap` walks a `std::map` of recordings in id order and stops at the cap. The cap is below the number of recordings, so the pass never reaches the later ones. Ids follow sorted filenames, `{digit}_{speaker}_{trial}`.
- *Evidence (real meta files; rid-to-file mapping confirmed with 0 valid-length mismatches over 2650 windows).*
  - FSDD fold 0: train has **digits 0 to 4 only**, validation **digits 0 to 2 only**, test all ten (m2, right). Half of the training recordings are never used.
  - EEGMMIDB fold 0: validation covers 11 of the 18 validation subjects.
  - AudioMNIST (inferred from code, never run; layout `NN/d_NN_t.wav`, so sorted speaker-first): 1000 recordings would be the first two training speakers of forty, validation the first 150 files of one speaker, test the first three speakers of the block. "Leave-one-group-out on 60 speakers" would train on about 2.
  - Effect on FSDD so far looks modest for PCA (0.707 to 0.703 with digit-balanced recordings). The AE gap moved from 0.082 to about 0.055 (1 to 2 seeds, different window draw, so not a clean contrast).
- *Satisfied by.* Sample recordings at random within group and stratify by digit and speaker. Print the composition per fold. Fix before the grid.
- *Verdict.* DISAGREE with C182 SUPPORTED. "Sampled round-robin across recordings" is literally true and materially misleading. The test `ShuffleDiversifiesKeptWindowIndex` checks window index, not recording coverage.

**V003 · FATAL · C025, C026, C234, C241, C292, C293, C294, C322 · "The cost is the saturation of the neuron's transfer function, not the spike."**
- *Objection.* AGREE with the Author that the sentence is not earned, but his reasons need correcting and a stronger one exists.
  1. **The dichotomy is false.** With direct input and one hidden layer, the time mean of a deterministic LIF is an elementwise function of the current, so the SNN-AE *is* a non-spiking network with activation f(I). There is no separate "spike" left to acquit. The transfer function is what the spike looks like after averaging.
  2. **That function is not a clip.** With hard reset under constant current, a neuron fires every k steps, so the output levels are 1, 1/2, 1/3... **At any T there is no level between 1/2 and 1** (m1). More steps refine only the low end. This strengthens C241, whose staircase counts suggest T removes the problem.
  3. **Saturation was never measured.** No figure or table in the paper shows how many neurons sit at 0 or 1.
- *Evidence from the experiment the Author could not run (encoder only, Xavier, 2 seeds, on saved windows):*

| | FSDD | excess | EEG | excess | Share of neuron×window exactly at 0 or 1 |
|---|---|---|---|---|---|
| Leaky ReLU | 0.667 | | 0.289 | | n/a |
| Sigmoid | 0.682 | +0.015 | 0.300 | +0.011 | n/a |
| Clamp [0,1] | 0.721 | +0.054 | 0.484 | +0.195 | 0.87 to 0.90 |
| Clamp, identity gradient | 0.764 | +0.097 | 0.537 | +0.248 | 0.95 to 0.99 |
| LIF T=4 | 0.727 | +0.060 | 0.442 | +0.153 | 0.90 (FSDD), 0.77 (EEG) |
| LIF T=16 | not run | | 0.397 | +0.108 | 0.54 (EEG) |

- *Reading, fairly.* This helps the Author more than his own dossier allows. Like for like, a sigmoid does stay near the control and a clamp does cost about as much as the SNN on FSDD. But it does not establish his sentence. On EEG the clamp is worse than the SNN at T=4 and much worse than at T=16. The trend follows the share of near-binary units, which falls as T rises, and a clip does not depend on T. A hard clamp trained this way collapses to a near-binary code, so saturation and discreteness are the same event here, not rivals. A smooth sigmoid is also bounded, so "saturation" alone cannot be the discriminator. All of this is direct input, and dev winners were mostly Poisson, where no such function exists.
- *Satisfied by.* Repeat this table in the harness or a logged twin, on header-free windows, with dispersion, and for a Poisson winner. Print the share of saturated units. Abstract, Discussion and Conclusion should say: "on a direct-coded, one-hidden-layer design, the gap tracks how close the hidden code is to binary." Figure F1.
- *Verdict.* AGREE UNSUPPORTED for C026, C292, C322. DISAGREE on three details. C294 is better than UNSUPPORTED (like-for-like support, though only direct input). Pressure-point item (e) on dead gradients is refuted: identity gradients make it *worse*, so a dead gradient is not the reason. C235/C236 remain WRONG as comparisons even though the conclusion on the sigmoid survives.

**V004 · FATAL · C007, C045, C046, C105, C002 · "Only" the neuron differs, "so the gap is the cost of spiking."**
- *Objection.* AGREE WRONG. Fig. 1 itself contradicts its caption, and Limitation (v) contradicts the abstract. The corrected drawing has five red blocks, including the two the paper never draws (m8).
- *Added.* The contribution also conflates what a spiking code can represent with what surrogate training finds. Nothing separates the two. Evidence that it matters: replacing the clamp's gradient alone moves error by 0.043 to 0.053 (V003 table), and the gap changes with budget (V014). Train-set error and a learning curve per arm would separate them.
- *Satisfied by.* P03 and P04, plus a second control that reads the spike code (the Author's own suggestion), or the claim limited to "the selected spiking pipeline". Mockup m8 replaces Fig. 1 at the same size.

### MAJOR

**V005 · MAJOR · C021, C057, C194, C195, C317 · Development-fold leakage is wider than the Author concedes.**
- The Author's fold-1 fix treats one speaker. On FSDD fold 0 *all six speakers* took part in the design study: george as test, jackson as validation, the rest as training (m2). So every headline test speaker informed the design, one directly and four through the models that were compared. FSDD cannot be cleaned by dropping a fold. Dropping fold 1 leaves 4 speakers.
- Undisclosed choices made on the same fold: the stopping skill 0.10, 1000 windows, the T range, and the surrogate sharpness, called "already best" from test MSE 0.754/0.734/0.744 (wiki, no validation figure).
- Only AudioMNIST and Siena are clean, and they are the two datasets never run, one of them hit by V002 and V017.
- *Satisfied by.* State that FSDD and EEGMMIDB are design-contaminated at speaker level. Treat clean-dataset results as primary confirmation, or keep a sealed set of EEGMMIDB subjects. Drawing F2.
- *Verdict.* AGREE PARTIAL, but P19 is too cheap.

**V006 · MAJOR · C288, C299, C300, C014 (conclusion on energy) · The energy outcome is largely fixed by the model and the normalization, and the paper presents it as an open empirical result.**
- *Evidence.* I re-implemented Table II including the spike path. It reproduces ANN [59,16] at 854.4853 nJ and the as-executed minus once difference of 11 017 nJ.
  - One synaptic event costs more than one MAC here: add + address + weight read + potential read and write, about 57 pJ against about 41 pJ. A spike-fed first layer beats the analog one only below about **0.5 to 0.6 input spikes per sample per window** (m3, right).
  - Latency coding gives exactly 1.0 by construction.
  - Poisson gives p·T, with p about 0.50 to 0.55 measured on real test and train windows, because per-window z-score plus min-max scaling makes silence as dense as speech. At the minimum T=2 that is already about 1.0.
  - Direct dense pays the same first layer plus the LIF. The only saving is in Lin(h→d), 14.7 nJ of 854 (1.7%).
  - So for a one-hidden-layer network, only the `recurrent` input transform can reach the winning region. The Discussion's "depends on the encoding, T and spike rate the search selects" hides this.
  - The Introduction's "a synapse whose input did not spike does no work" cannot apply: the normalization removes the input sparsity.
- *Satisfied by.* Say it, and plot it. A sparse input, for example one without per-window scaling, would be a fair test of the spiking claim. Figure F3. Caveat: the example is widths [59,16]; deeper winners add spike-fed layers, subject to a similar bound on ρ·T.
- *Verdict.* DISAGREE with C300 SUPPORTED as phrased.

**V007 · MAJOR · new; C031, C158, Table IV · Matched widths are not matched quality, and the shared decoder dilutes the ratio.**
- The ANN-AE is matched in width, so the two arms sit at different MSE. The relevant question for "what does the spike return" is the energy at equal error. A narrower ANN would reach the SNN's error for less, so the reported ratio favours the SNN. No sweep exists.
- The decoder is 430 of 854 nJ and identical in both arms. Whole-network ratio 1.53× for the T=28 winner, encoder-only 2.06×. That encoder is the on-device part in the paper's own use case (C031, AGREE).
- Table IV reports means over heterogeneous winners: 1304 and 3875 nJ average to 2590. A ratio of means hides that bimodality.
- *Satisfied by.* An ANN width sweep as a curve in Fig. 2, encoder-only and total columns, and each winner plotted individually.

**V008 · MAJOR · C013, C204–C206 · "Energy first" is close to empty.**
- AGREE WRONG, and PCA pricing is the strongest point. Added: the rule's first branch is effectively unreachable. With 500 test recordings per FSDD speaker (2500 over headline folds) and pairing, nearly any gap will exclude zero, so the rule falls through to "report both". Energy is also absent from selection (the winner is lowest MSE). "The cost we put first" therefore governs nothing that I can find. Dev folds suggest neither branch would favour the SNN anyway.
- *Satisfied by.* P21 with a stated margin, and PCA costed. Or drop the rhetoric.

**V009 · MAJOR · C110, C111, C232, C233, C151 · "Favours", "conservative", "best-against-best".**
- AGREE with the Author's direction table. Added: the control itself is under-trained. On dev folds it loses to PCA, which is linear, on EEG at 0.291 vs 0.183, and matches it on clean FSDD (V001). A nonlinear AE that cannot match a linear one at 4:1 is not at its best. So the gap is measured against a soft floor, and nothing bounds what a better control would reach.
- "Best-against-best" tests two initializers, neither designed for LIF neurons; data-driven SNN initialization exists (Rossbroich et al., 2022). A single knob moved the EEG gap from about 0 to 0.156, so the gap is fragile. It should be reported across initializers, not fixed at the one that makes the story tidy.
- *Verdict.* C233 should be UNSUPPORTED rather than PARTIAL.

**V010 · MAJOR · C012, C198–C203, C302, C303 · The primary test does not answer a leave-one-speaker question.**
- AGREE with the Author, including that group-level results are promised and not printed (C202 confirmed by reading `04_`). Added: FSDD test recordings carry 3 windows, EEG 5 to 6, train 1. Averaging seeds before testing removes between-search variance, the largest source according to the wiki ("close to a coin flip" at 8 evaluations). The CI is therefore conditional on the selected architectures. Holm's family includes Mean, a trivial comparison.
- *Satisfied by.* A per-group forest plot (F7) as the primary display, with the recording test secondary.

**V011 · MAJOR · C272, C273, C276, C277 · T1's FSDD numbers mean something other than they say.**
- The figures recompute correctly (AGREE). Their meaning does not. On FSDD a "plateau" Transformer errs 0.113 on window 0 and 1.000 on all other windows, against 1.007 for the mean. Of its windows 10% below the mean predictor, 95% are window 0. So "0.935 against 0.996" is the WAV header being memorized, and the "8% of windows" in the notes are those windows. Only the armed fit at 0.867 shows real gain (0.9305 on other windows).
- The verdict also mixes denominators (8 trainings, 4 final fits), and shows no PCA or AE reference, so a reader cannot see that the best fit is nowhere near them (m5).
- *Verdict.* DISAGREE with C276's "sub-threshold learning" for FSDD. It holds on EEG only. C277 AGREE. Rerun after V001, on the format-9 winners.

**V012 · MAJOR · C242, C291, C297, new · The design tests a rate code only, and this limit is not stated.**
- After the latent mean, output spike order is irrelevant: shuffling each neuron's spike times leaves z̄ unchanged (m4). Any timing a temporal code could carry is discarded by construction. Conclusions about "the spike" apply to rate-read spiking, and the Related Work cites temporal coding as the prior success (Comşa).
- *Satisfied by.* State it as a limitation, in the abstract's scope. Small inset, F8.

**V013 · MAJOR · C077, C102, C174, C185, new · Undisclosed or misdescribed data handling.**
- Confirmed: 36 of 1526 EDF files at 128 Hz, subjects S088, S092, S100 (AGREE C185; the profile comment is false).
- EEG band-pass and notch are absent from the paper.
- New: per-recording caps keep the *first* windows (`source_window_index >= cap`). EEGMMIDB therefore keeps the first 64 s of each run, and Siena the first 20 s of each file, in a seizure database where "(epilepsy)" suggests otherwise. The paper says only "capped".

**V014 · MAJOR · C046, C313, new · The gap is a budget-dependent number and the abstract does not say so.**
- Wiki table: the gap runs 0.165 at 1000 windows to 0.220 at 10 000, with SNN error flat from 3000 on. In the probe every EEG arm's best epoch is 19 to 30 of 30, most at or near the cap. 1000 windows was chosen on the dev fold. The paper discloses the epoch cap (vii) and not the window dependence.
- *Satisfied by.* Learning curves per arm and the windows sweep (F6).

**V015 · MAJOR · C016, C280, Figs 1 and 2 · Visual audit failure as a whole.**
- The paper has 2 figures, one a placeholder. None of the eight key mechanisms is visible (Section 2). The argument of Section V is a table and a paragraph of paired numbers that switch comparator, which is exactly how C235 and C236 survived. In m6, plotted as dots, the mismatch of initializers is obvious at a glance: every clamp and sigmoid point has no Xavier partner.
- Fig. 2 issues: y error bars only, though E varies by winner far more than MSE; a single mean point per model; PCA as a line only because it was not costed; log axis with no ANN curve. The header comment of `fig_pareto.tex` still says arithmetic energy, while the CSV is memory-aware.

### MINOR

**V016 · MINOR · C092, C088, C120, C125, C170, C173, C240 · Factual slips.** I re-confirmed C092 (reset factor), C088, C125, C170, C173 and C240 in the code or by simulation. AGREE all WRONG. C120 taken from `checks.py` logic, not rerun.

**V017 · MINOR (MAJOR if AudioMNIST stays) · C184, new · AudioMNIST windows.** The cap of 2 keeps windows 0 and 1, so about half the kept windows would be header-dominated if V001 stands. The wiki's explanation of the 2026-09 degeneracy (a silent lead-in, 1 unique spike pattern in 2700) deserves re-examination: the header is a likelier cause. Hypothesis, unverified: no AudioMNIST on this machine. `check_degenerate_reconstruction`'s threshold of 1e-4 would not catch an error of about 0.1. The "near-silent lead-in" finding also means many remaining windows are amplified silence, which is not "short speech".

**V018 · MINOR · C010, C047 · Naming.** Six blocks of up to 18 subjects is grouped 6-fold, and one validation block is nested hold-out. "Leave-one-group-out" is true only for FSDD.

**V019 · MINOR · C116, C118, C127 · Thin selection.** 48 evaluations over 7 genes, on 150 validation windows of one block (digits 0 to 2 on FSDD). Table V will be read as what spiking "prefers". Report agreement across seeds, and do not interpret modal genes. Mean re-scoring includes the selected-for score (AGREE).

**V020 · MINOR · new · Trained neuron parameters are not reported.** Table V prints initial V_th and β, and the argument in Section V uses initial values. In my replica V_th moved from 0.6 to about 0.8 and from 0.96 to about 2.2 to 2.6, and β ran to its upper clamp. Replica parametrization differs, so treat as a prompt: print the trained values, because the transfer function being argued about is the trained one.

**V021 · MINOR · C144, C053, C148, C151, C308 · Energy wording.** AGREE. Lemaire's text confirmed: "the whole processing must be repeated", "between 6.25 and 8.02 times lower", 0.1 and 3.1 pJ integer prices. Horowitz's 64-bit cache figure applied to 32-bit words must be stated.

**V022 · MINOR · C003, C316, C315, C319, C252, C018, C311 · Inflation and contradictions.** All AGREE. Conclusion "every choice made on validation groups" contradicts Section V. Past tense for runs that have not happened. Note also that the abstract says "a design study ... shows" for evidence Section V calls "argued rather than measured".

---

## 2. Visual audit

| Mechanism | Can a reader SEE it? | Demand |
|---|---|---|
| LIF transfer function, saturation | No. One inequality in prose | F1 |
| LOSO layout, dev-fold contamination, caps | No. Prose only | F2 |
| What differs between SNN-AE and ANN-AE | Misleading. Fig. 1 omits two blocks, caption says one | F4 |
| Energy accounting | No. Table of symbols, results table placeholder | F3 |
| First-layer reuse under direct coding | No. Prose only | F3, one bar pair |
| T1 arms against the plateau | No. A 70-word macro sentence | F5 |
| Latent time mean | Partly. A red box, with no consequence shown | F8 |
| Selection pipeline | No | F9 |

## 3. Figure demands, ranked

Budget: about 48 cm of column is one page. Items F1 to F5 total about 18 cm net of the table replaced.

| # | Figure | Panels and axes | Data | Column space |
|---|---|---|---|---|
| F1 | **LIF is a ladder, not a clip** (m1) | Output vs constant current for LIF T=4, T→∞, clamp, sigmoid; add a second panel of share of units at 0 or 1 vs T | Left exists (pure maths). Right needs logging, small twin run | 1 col, 4.5 cm |
| F2 | **Fold map and what the caps keep** (m2) | Speakers × folds grid with dev rows marked; digits × speakers coverage after the fix | Exists | 1 col, 4.5 cm |
| F3 | **Where the energy goes, and when a spike wins** (m3) | Stacked bars per block: ANN, direct once, direct as executed, Poisson; break-even curve vs θ/M | Maths now, real winners after the run | 1 col, 4.5 cm |
| F4 | **Corrected Fig. 1** (m8) | Five red blocks | Exists | replaces Fig. 1, 0 net |
| F5 | **Design study as dots, and T1 strip** (m6, m5) | Dots replace Table III; strip replaces the T1 sentence's numbers | Exists, rerun after V001 | replaces a table, about 3 cm net |
| F6 | Learning curves per arm, windows sweep | MSE vs epoch; gap vs training windows | Events logs after the run; sweep needs rerun | 1 col, 4 cm |
| F7 | Per-group forest plot | d per speaker or subject, per dataset | After the run | 1 col, 5 cm |
| F8 | Time-mean inset (m4) | Raster, shuffled raster, same latent | Exists | inset, 2.5 cm |
| F9 | Selection pipeline (m9) | One outer fold; where dev decisions enter | Exists | 1 col, 3.5 cm; drop first if space is short |

Also change Fig. 2: individual winners as points with an ANN width-sweep curve and a PCA point once costed.

---

## 4. Verified and sound

**Author's SUPPORTED claims I checked and accept** (code, recomputation or source):
- Model and training: C084, C085 (reset, strict threshold); C089, C090; C091 (surrogate, used in the replica); C093 to C096; C106 to C109 (twin config read); C137, C138; C124.
- Splits: C187 to C191 (fold rule, LEAKAGE abort lines 1452 and 1459); C189 from the manifest; C195; C197; C049; C180 (windows do not overlap, up to the shift of V001).
- Energy: C149, C150; C152; C154; C155; C157; C158 (11 017 nJ reproduced); C160; C163 to C169 (all reproduced by the independent implementation); C146 prices; C070.
- T1 and statistics: C272 to C275 as arithmetic (recomputed 0.9162, 0.9305, 0.9350; 0.9934, 0.9210, 0.9925); C304 (0.0625).
- Others: C177 and C185's count (1526 headers read: 1490 at 160, 36 at 128); C183.

**Table III** as a set is more credible than its "PARTIAL, no logs" status suggests: the replica reproduces the Xavier rows on saved windows within about 0.01. The numbers still carry V001 on FSDD.

**Credit.** The dossier is unusually candid, and its strongest concessions hold. The harness's split assertion, coverage checks and generated-table discipline are sound engineering. T1's sentence draws no conclusion it cannot support. Limitations (i) and (iii) are honest.

**Author's statuses I dispute.** C182, C300 (should not be SUPPORTED); C114, C133, C273 (FSDD meaning); C276; C233 (should be UNSUPPORTED); C294 (better than UNSUPPORTED); pressure point 3(e).

## 5. Could not verify

- AudioMNIST and Siena: not on this machine. V002's AudioMNIST consequence and V017 are code inference.
- Harness runs: no build, no C++ tests. The probe is a replica, with 2 seeds, 1 seed on header-free retraining; Poisson winners not probed; effect of V002 on the AE gap not cleanly isolated (the balanced run confounds the window draw, and the fixed-order arm crashed on a shared archive).
- Whether the header affects other wave-reading experiments: outside this review.
- Dev job numbers (0.713, 0.661, 0.407, 0.291, 0.183, 12 321 nJ) come from QT; the PCA 0.649 is reproduced, 0.183 was not recomputed. Design-study logs do not exist.
- One diagnostic (`loc.py`, full window-bank search) was blocked by a PreToolUse hook. I did not bypass it; the alignment check on five windows and the header bytes settled the question another way. Main-grid outcomes, pre-validation job 5316016, Zenodo contents, and paywalled Dampfhoffer and Yan full texts were not checked.
- Reads outside the snapshot, all read-only: FSDD and EEGMMIDB files, saved `.npy` windows and `FsddWindowDataset.cpp`, which is missing from the snapshot.

## 6. Verdict: MAJOR REVISION

Not ACCEPT or MINOR: results are placeholders, the core claim is unsupported as written, and speech data are defective. Not a flat REJECT: the design is sound in intent, the defects were found before the grid, and the fixes are cheap now and very expensive after a months-long run.

Required before the grid: V001, V002, V013 (128 Hz), and the probe of V003 in logged form. Required before submission: V003, V004, V005, V006, V007, V008, F1 to F5. If submitted without V001 and V002 fixed, my verdict is REJECT.

Sources: [Rossbroich et al., arXiv 2206.10226](https://arxiv.org/abs/2206.10226v1); [Lemaire et al., arXiv 2210.13107](https://arxiv.org/pdf/2210.13107).
