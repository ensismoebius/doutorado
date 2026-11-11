# Reviewer 1 ruling on the Author's response (Round 3)

Scope: `author_response.md` and `response_scripts/`. I did not open `review_math.md` or `ruling_math.md`, so I rule on m23, m25 and m26 from the Author's account of them only.

**Verdict: MAJOR REVISION.** REJECT if the grid starts or the paper is submitted before the conditions in Section 6 hold.

The Author's response is strong: most concessions are correct, and he caught real errors of mine. But four of his new claims fail when tested, including the two he offered against his own side. New runs are in `visual_mockups/probe_src/` (`p3.py`, `p3.log`, `q.py`). They use the same replica, on clean FSDD fold 0, with 4 seeds where stated.

## 1. What I got wrong

- **The 0.082 clean gap: withdrawn.** It was a high draw. Four seeds on clean FSDD give ANN-AE 0.7231 ± 0.0111, SNN T=4 0.7895 ± 0.0065, gap 0.066. The Author's 0.064 stands.
- **The 41 and 57 pJ prices: withdrawn.** I sized the first-layer weight memory wrongly. It is 15 163 words, about 59 KB, so 22.2 pJ per read. His 26.8 pJ (MAC plus weight) and 43.2 pJ (event) reproduce. His 27.1 pJ also includes input reads. The true ratio, 1.6, is worse for the spike than mine was.
- **Break-even 0.5 to 0.6: corrected.** My curve left out the Poisson generator. With it, the range is 0.36 to 0.60, falling with T. His figures reproduce.
- **Poisson p 0.50 to 0.55: corrected.** Clean test windows give 0.503, EEG 0.437. Train alone reached my 0.547, so the range was a poor summary.
- **V014's framing: conceded.** The windows table is format 4 and not monotone. It does not measure the adopted design.

## 2. Tests of the Author's new claims

| Claim | Result | Ruling |
|---|---|---|
| Clamp on clean FSDD is worse than the SNN (0.795 vs 0.779, one seed) | 4 seeds: clamp 0.7963 ± 0.0087, SNN T=4 0.7895 ± 0.0065; difference 0.007, SE 0.005; T=16 0.7977 (2 seeds) | **Not established.** They tie |
| "Partly the header" explains clamp matching SNN | On clean FSDD excess is +0.073 for clamp, +0.066 for LIF T=4 | **Rejected.** The match survives without the header |
| Decoder effect −0.005 / +0.003 | Arithmetic right, comparison wrong: those rows also change the pooling point | **Rejected** (N3) |
| Trained sigmoid is not saturated | Reproduced at the ≤0.01/≥0.99 cut (FSDD 3.9%, EEG 0.3%). At ≤0.1/≥0.9, FSDD is 31% | **Stands, with the cut stated.** It is threshold-dependent, and still far below LIF |
| Excess rises with the pinned share (Q3) | EEG yes. FSDD no: LIF T=4 90%/+0.066, LIF T=16 87%/+0.075, clamp 86%/+0.073 | **Rejected as a general statement** |
| Header shifts gap by about 0.004 | 0.060 → 0.066 on 4 seeds, same size | Stands |
| ICC 0.33, 19 of 19 subjects, SE 2.3× | Reproduced from his script | Stands, one pair and one fold |
| V006 script, M5 table | First rows reproduced | Stands |
| Which 1000 windows moves PCA k=16 by 0.047 | 24 random draws: mean 0.686, sd 0.016, range 0.661 to 0.716. His 0.648 lies below my range | Direction stands, size overstated. The sd is 0.016 |
| AudioMNIST port | Logic read; I agree with it | Stands, as a port |

Also, the sigmoid's corrected excess on clean FSDD is +0.020 on 4 seeds, not +0.034.

**Side finding.** Replica runs with the same seed and windows differ between his run and mine: LIF 0.7734 vs 0.7924 at seed 41, while clamp and sigmoid match to six digits. I did not find the cause. Single-seed replica numbers are therefore weaker than their digits suggest, and this applies to both of us.

## 3. Ruling on my objections

| ID | Ruling | Reason |
|---|---|---|
| V001 | SUSTAINED | Confirmed in full. Only my 0.082 is withdrawn |
| V002 | MODIFIED | Blocker and fatal for AudioMNIST. FSDD effect is within draw noise, so not fatal there |
| V003 | SUSTAINED | Sentence stays unsupported, and his replacement text overclaims (N4) |
| V004 | SUSTAINED | Concession stands, but my train-set-error demand is unanswered |
| V005 | SUSTAINED | Conceded. His block-0-validates-fold-5 addition is correct |
| V006 | MODIFIED | My prices are withdrawn; conclusion stands, and I now verify the `recurrent` exception (N5) |
| V007 | SUSTAINED | Conceded; a width sweep "only if it can be run" leaves the matched-quality point open |
| V008 | SUSTAINED | Conceded, but line 51 "energy is the cost we put first" survives Q1 |
| V009 | SUSTAINED | Conceded |
| V010 | SUSTAINED | Conceded; ICC strengthens it |
| V011 | SUSTAINED | Conceded; 11 of 12 fits decompose as I said |
| V012 | SUSTAINED | Conceded; Q6 is enough with an m8 annotation |
| V013 | SUSTAINED | D3 and D4 confirmed by code; 36 files re-read |
| V014 | MODIFIED | Format-4 evidence only; dependence on budget remains unmeasured for the adopted design |
| V015 | SUSTAINED | Figures accepted, with changes in Section 5 |
| V016 | SUSTAINED | Conceded |
| V017 | MODIFIED | Stays hypothesis. His "could not find" is closed: `02_...py:500`, threshold 1e-4 and 2%, so an error near 0.1 passes |
| V018 | SUSTAINED | Conceded; Q12 is incomplete (N8) |
| V019 | SUSTAINED | Conceded |
| V020 | SUSTAINED | Conceded |
| V021 | SUSTAINED | Conceded |
| V022 | SUSTAINED | Conceded |

Nothing is withdrawn whole. Two numbers and one framing are withdrawn inside V001, V006 and V014.

**The three REBUTs**
- **m23: rebuttal holds.** The profile exists and its ranges match the macros; I read it in Round 1.
- **m25: rebuttal holds.** Paper lines 279 to 280 attribute the add to nobody, and Lemaire's extracted text has no threshold term.
- **m26: rebuttal holds for test rows.** The comment at `Meeting01Experiment.cpp:535` confirms they are recomputed. It says nothing about GA checkpoints (N11).

## 4. D1 to D9 and job 5316016

- **D1: adequate fix.** Add a check of the `fmt ` chunk too. Confirm AudioMNIST's `sox` headers on real files.
- **D2: adequate in design, consequence missing.** A seed-varied draw breaks "PCA and Mean are fitted once" and must keep test windows fixed (N10).
- **D3: partly disagree.** "Sample inside the voiced region" revives the voice-activity filtering the wiki rejected as a selection risk. Prefer dropping the cap or uniform sampling. Silent-window amplification remains unaddressed.
- **D4: fix as written is unsafe** (N2).
- **D5: agree.**
- **D6: agree.** Specify the interval for 4 to 5 speakers.
- **D7: agree, one wording fix.** "Test numbers unaffected" holds only in distribution. It also adds a training per front member.
- **D8: agree.**
- **D9: agree.**
- **Missing:** stopping threshold recalibration (N1) and checkpoint invalidation (N11).

**Job 5316016.** I agree on FSDD: not printable, windows ≥ 1 are fit for the go/no-go, and armed flags are void. I would add that clean runs may arm later, so its FSDD wall times probably understate clean cost.

I disagree in part on EEGMMIDB. It is usable for the removal decision. It is not usable as a printed row beside a format-10 grid, because paper line 453 says the pre-validation uses "the windows, training and selection" of the main sections, and after D2 to D4 that is false for EEG too. Either rerun EEG with FSDD, or rewrite that sentence and disclose it. The Author's premises about format and per-window CSVs remain unchecked.

## 5. New objections, and text and figures

**N1 · MAJOR · The stopping threshold rests on the header.** `Meeting01Stopping.hpp:33-37` justifies s = 0.10 by an FSDD plateau of 0.962 and a deepest excursion of 0.066. Both are header-sized. The Author notes the skewed arming but keeps 0.10. It must be recalibrated on clean data, using validation only.

**N2 · MAJOR · Excluding S088, S092, S100 before folding moves the blocks.** With `rank*6/n`, n = 106 moves 7 subjects. **S019 goes from development test block 0 to block 1, the test block of headline fold 1**, and S037 leaves the development validation block. Assign blocks on all 109 subjects, then drop the three.

**N3 · MAJOR · Q7, Q1 and the Part D removal reverse a real effect.** −0.005/+0.003 changes the pooling point as well as the decoder. The one-factor contrast, same output mean at T=16 under Kaiming, is in the wiki and the Author's own C022: FSDD 0.734 → 0.721, EEG 0.454 → 0.403. Replacing the spiking decoder *lowered* error by 0.013 and 0.051. Latent versus output pooling then costs 0.008 and 0.054. "Left error unchanged" is unsupported, and M19's "worse" does not follow. The numbers need regeneration and crossing, per M18. No Xavier equivalent exists.

**N4 · MAJOR · Q1, Q3 and Q17 introduce an unsupported claim.** My own V003 wording helped cause this, and I had conditioned it on a harness rerun.
- "A clamp is worse than the SNN on both datasets, so a clip does not reproduce the gap" fails on FSDD (tie).
- "The excess rising with that share" fails on FSDD.
- The two-sided share is also a poor measure. The leaky-ReLU control itself has 49% of values at ≤ 0.01 on FSDD with no excess, so the difference lies in the upper values. Print each side.
- "Tracks" and "tied to" in an abstract and conclusion rest on a replica, direct input, one genome per dataset.

Safer: "We did not determine what causes the remaining gap; on direct-coded networks a clamp cost about as much as the SNN on speech and more on EEG, and a sigmoid cost little."

**N5 · MAJOR · Q5's bounds are false as written.**
- "Direct coding saves at most [0.3]%" is one point: [59,16], T=2, ρ=0.1. At ρ=0.01 it is 1.0%. At [65,64] it is 3.4 to 5.8%, and at [128,64] 4.7 to 7.1%. EEG widths matter because the spike-fed latent layer is larger.
- "A spike-fed first layer costs more than an analog one" fails for `recurrent`. On real windows, direct plus recurrent gives θ/M about 0.24 at T=2 (V_th 1.5, β 0.8) and 0.58 at T=4, with a total ratio near 0.8 at the low end by my ledger (ρ assumed). The winning region exists but is narrow.

**N6 · MINOR · Q9.** "No test can reach 0.05" is wrong: rank tests cannot, a t-test can, and his own simulation uses one. Δ must be a number before the grid.

**N7 · MINOR · Q11.** "AudioMNIST and Siena were not used in design" overstates. AudioMNIST rows were read in September and drove the cap fix (wiki line 272). Say "not used in the design study".

**N8 · MINOR · Text left behind.** Q12 misses abstract line 50 and Section line 352. Q1 keeps line 51. The title's "Leakage-Controlled" needs review after V005.

**N9 · MINOR · Q15.** "Does not change its side of 1" was tested on five points, none near 1. Direct T=2 sits at 0.997.

**N10 · MINOR · D2 consequence.** Seed-varied draws need PCA and Mean per draw, a fixed test set, and changes to `03_` and `check_coverage`.

**N11 · MAJOR · Resume after D1 to D3.** `config_hash` hashes the resolved configuration, and a code fix leaves that unchanged. The Author says the format bump prevents mixing, but I did not find format in the GA checkpoint code and could not confirm it. One later grep was blocked by a hook, which I did not bypass. Delete old checkpoints or add a data-version field, and test it.

**Other Q text.** Q2, Q4, Q6, Q8, Q10, Q13, Q14 and Q16 are acceptable once brackets are filled from logged runs. Q3's staircase sentence is correct.

**Accepted figures.**
- m8: satisfied. Add a note on the mean box that timing is discarded; that settles F8.
- F1: satisfied if the right panel shows each side separately, and for both datasets.
- F2: satisfied, showing the pre-fix and post-fix composition.
- F3: satisfied only with the corrected curve, recurrent marked, and EEG widths shown.
- F5: satisfied with dispersion, and T1 regenerated. The m5 mockup carries the header and must not be reused.
- F7: satisfied.
- F9: declining is fine.
- F6: partly refused. The window sweep can go, but per-arm learning curves and train-set error are free from event logs, and they are the only evidence on V004's split between representation and training.

## 6. Conditions

**Before the main grid**
1. D1, D2, D3, and D4 with blocks assigned on 109 subjects (N2).
2. Tests: WAV known-answer, recording coverage, fold composition.
3. AudioMNIST and Siena verified on real files, or dropped. Confirm window content, not only counts.
4. s recalibrated on clean data (N1).
5. GA checkpoints invalidated (N11); results format bumped.
6. Margin Δ, primary group-level estimand and encoder-only and PCA energy fixed in writing. D5, D6.
7. Decision on seed-varied window draw (N10), and D7.
8. Logging of per-side hidden value shares, trained V_th and β, and training error.
9. Stale design study and T1 regenerated on clean data, or removed.

**Before submission**
1. The saturation sentence gone, and Q1, Q3, Q5, Q7, Q9, Q17 rewritten per N3 to N6.
2. The clamp and sigmoid comparison in a logged twin with at least 4 seeds, crossed with decoder and pooling, including a Poisson winner.
3. All V and N text fixes applied, Section 5 figures produced, and placeholders replaced.
4. Pre-validation rerun or its sentence rewritten.
5. FSDD and EEGMMIDB labelled design-informed.
6. Disclosure that earlier speech numbers carried D1.

## 7. Could not verify

Harness behaviour after fixes; AudioMNIST and Siena; job 5316016; GA checkpoint keying; the cause of the replica's run-to-run difference; EEG sigmoid share at the 0.1 cut; the texts of M- and m-series objections.
