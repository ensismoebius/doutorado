# Experiment05: Biometric Authentication of Severely Dysphonic Speakers via Imagined Speech

> **Thesis primary experiment.**  
> Thesis: *Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined Speech*  
> Author: André Furlan — UNESP | Advisor: Prof. Dr. Rodrigo Capobianco Guido | Funding: FAPESP 2021/12407-4

---

## Overview

Experiment05 implements the full speaker-authentication pipeline for individuals with severe laryngeal dysphonia (DLS). The core hypothesis: combining a degraded phonated voice with EEG-captured imagined speech produces a biometric signal that is both robust (works even when voice quality degrades) and difficult to spoof (EEG is not externally recordable without consent).

The experiment consists of two stages:

- **E3 — Feature Extraction**: handcrafted (DTWPT + classical descriptors, guided by paraconsistent quality ranking) vs. learned (SNN-AE, ANN-AE) and PCA, a classic linear feature extractor used as the reference the learned ones must beat (implemented 2026-11-11, not yet run; LSTM-AE pre-validated and dropped, see `Research-Context.md`)
- **E4 — Authentication**: the DSNN classifier, against two controls on the same features and folds (its non-spiking twin `ann`, and `cosine`, no network); test phrases seen or unseen in training; nested or flat 5-fold cross-validation; simulated dysphonia; measured inference energy (see [Protocol revision](#protocol-revision-2026-11-12))

> **Status (2026-11-12): every thesis number so far is void; the revised protocol below is
> implemented and tested, and both phases must run again.** The July run had four defects
> (`base` = `small`, an inverted EER, a fold intersection that dropped 62% of the training
> rows, a nested text leak; fixed b22f07c1). The re-run of 2026-11-12 fixed those, but its EEG
> was the *previous* trial's EEG ([below](#1-the-eeg-was-the-previous-trials-eeg)), so its EEG
> and fusion numbers are void too; only its voice numbers stand, and they are superseded by the
> new text split.

> **Follow-up search:** because the 24 hand-picked autoencoder profiles lost to
> handcrafted features on `D_penalized`, the [paraconsistentGA](ParaconsistentGA.md)
> experiment searches the same architecture axes automatically with NSGA-II, to test
> whether a better autoencoder exists than those fixed phase00 tiers.

---

## Protocol revision (2026-11-12)

A review of the thesis (`documentation/00-thesis/monography/issues.md`, 21 items) found that
the experiment could not answer several of the questions the text asks. Each change below
starts from the question it lets the experiment answer.

### 1. The EEG was the previous trial's EEG

**Problem.** Fusion and the EEG results rest on one assumption: the EEG row paired with a voice
row is *the same trial* — same speaker, same word, same moment. The loader read
`eeg_samples` at `original_row - 1`, believing the rows 1-based. They are 0-based, so every
voice trial was paired with the trial recorded **before** it:

```
 voice row (original_row)   EEG the old loader read      what that EEG trial was
 ------------------------   ------------------------     -----------------------------------
 subject 3, row 120 "/a/"   subject 3, row 119           often an IMAGINED trial, another word
```

Over the 1974 voice trials: 1833 EEG rows came from *pronounced* trials and 141 from
*imagined* ones, and only 193 of 1974 had the voice trial's word. **Silent:** every row had
EEG of the right shape and the right speaker, so nothing failed; the speaker information in the
EEG survived (same subject), the word did not.

**Fix (`EegPairing.hpp`, `ThesisDataset.cpp`).** Pairing is by trial identity: the EEG row
must have the voice trial's `original_row` and modality *Pronounced*, or the load throws.
Identity is right, not another guess: the trials that carry audio are exactly the 1974
Pronounced trials and none of the Imagined ones, which a one-row shift would break (the
shared core loaders had the same shift and were fixed too, see
[DataLoaders](../Core/DataLoaders.md#audio--eeg-pairing-the-label-is-the-0-based-eeg-row-of-the-same-trial-fixed-2026-11-12)). The 10.1117 corpus also has 7341 *imagined* trials (EEG only, ~3.7 per voice
trial, every (speaker, word) covered), so the EEG condition became an experimental factor:

| `dataset.eeg_condition` | EEG row of voice trial (speaker s, word w) | Question it answers |
|---|---|---|
| `pronounced` | the same trial (spoken aloud, recorded together) | EEG of overt speech |
| `imagined` | an imagined trial of (s, w), drawn without replacement with `eeg_pairing_seed` | the thesis title: imagined speech |

The cache name carries both (`~/database.sqlite.pronounced.e05dscache`,
`~/database.sqlite.imagined-2026.e05dscache`, format version 3); the old
`~/database.sqlite.e05dscache` holds the defective pairing and is read by nothing.
Tests: `ThesisDatasetPairing.*` build a SQLite with the real schema whose EEG value encodes
`1000·subject + original_row`, so the test reads *which* trial arrived.

### 2. Seen and unseen phrases on the same training set

**Problem.** "Text-dependent" trained on 80% of the phrases and "text-independent" on ~45%, so
the difference between the modes mixed *seeing the phrase* with *having twice the training
data* (issue 4), and the thesis used two definitions of the term (issue 9).

**Fix (`make_text_split`).** The 11 phrases are shuffled with the seed; the first 5 train in
**both** modes; the test phrases are the same 5 (seen) or the next 5 (unseen):

```
 phrases (shuffled)   P1 P2 P3 P4 P5 | P6 P7 P8 P9 P10 | P11
 training, both modes ############## |                 |
 test, "dependent"    ############## |                 |       seen phrases
 test, "independent"                 | ############### |       unseen phrases
```

Speakers stay disjoint across folds (GroupKFold) in both. The table labels are "frases
vistas" / "frases não vistas", the only definition the thesis uses now.

### 3. Controls: what does spiking buy, what does a network buy?

**Problem.** An EER alone cannot say whether the DSNN contributes anything (issue 6). The old
non-spiking control (`rnn` = `SimpleResNet`) was not a twin: for `residual:2` it built 4 hidden
128×128 matrices where the DSNN builds 1.

| `classifier.type` | network | answers |
|---|---|---|
| `dsnn` | Linear → LIF → (Linear → LIF + skip) × (D − 1) → Linear, T = 16 | the thesis classifier |
| `ann` | the same graph, ReLU for LIF, no time steps, **the same initial weights** | what spiking costs/buys |
| `cosine` | none: the standardized features are the embedding (flat CV only) | what any network buys |

`AnnTwin.StartsFromTheDsnnWeights` checks the shared initialization.

### 4. Simulated dysphonia

**Problem.** The title promises authentication of dysphonic speakers; no dysphonic voice was
ever tested (issue 1). **Fix (`VoiceDegradation.hpp`, `degradation` block).** A whispered
version of each voice is synthesized (LPC order 2 + fs/1000, 25 ms frames: the vocal-tract
filter of each frame driven by white noise instead of the glottal pulses) and mixed in:
`y = √(1−m)·x + √m·w`, energy-preserving, `m = degradation.level ∈ (0, 1]`. `scope`:
`probes` keeps each test speaker's enrolment utterance clean (a speaker enrolled before the
illness), `test` degrades every test utterance. Training is always clean, and the scaler is
fitted on clean rows. Tests check the power (±20%), that periodicity drops from > 0.8 to < 0.3,
and that the effect grows with `m`. It is a model of breathy/aphonic voice, not of every
dysphonia: the thesis says so.

### 5. How stable is the Phase 00 winner?

**Problem.** The extractor was chosen with every speaker's labels, the same speakers Phase 01
tests on (issue 5), and a deterministic extractor's three repetitions are one number, so
σ = 0 said nothing (issue 8). **Fix (`ParaconsistentSubsets.hpp`).** Every Phase 00 profile
also scores its extractor on 30 speaker subsets: the 15 leave-one-speaker-out sets and the
training speakers of each of the 15 Phase 01 outer folds (k = 5, seeds 42–44). The ranking
script reports, per signal, in how many fold subsets the full-set winner is still the winner
(`winner_stability.csv`), and the tables print `loso_sd` and `fold_wins`.

### 6. Measured inference energy

**Problem.** The thesis argues for an SNN on energy and never measured it (issue 11). **Fix.**
Each network fold counts its **test pass**: spikes per layer, and from them Meeting01's
memory-aware ledger (Lemaire et al. 2022 at Horowitz 2014 prices, `Meeting01Energy.hpp`, the
same model as the meeting01 paper). Written per fold and as a mean under `energy` in the
summary: `nj` (primary: the repeated input's first layer computed once), `nj_as_executed`
(computed at every step), `nj_sram8k/32k/1m` and `ops_nj` (checks). The extractor is not
included: it is the same for every classifier of one feature set. The residual sums are priced
as event routing: `W(a + b) = Wa + Wb`, so a layer fed by `h = Σ spike trains` costs `sum(h)`
events, each earlier spike routed again. Before, `final_sops` came from the last *training*
batch and ignored the skips.

### 7. Statistics

`thesis_cross_profile_significance.py --pairing corrected` is the Nadeau–Bengio corrected
repeated k-fold t-test: 3 repetitions × 5 folds are 15 correlated values, not 15 independent
ones. On the (void) 2026-11-12 EEG/fusion numbers it moved late fusion vs EEG from
p = 0.035 (Wilcoxon) to p = 0.27. `--select` (required) declares the family compared, and the
reference should be fixed in advance (`--reference <run_tag>`).

### 8. The profiles are generated

`scripts/pipeline/thesis/00_thesis_generate_profiles.py` writes every profile (275 Phase 00 =
153 voice + 61 × 2 EEG conditions; 142 Phase 01 = grid 56 + controls 28 + dysphonia 40 +
extractor comparison 18) and `profiles/profiles.manifest`, which
`thesis_profile_audit_gtest` audits; a profile on disk that the manifest does not name fails.
It reproduced the 246 earlier profiles exactly (checked, apart from the new keys).

### 9. What the autoencoders and PCA read: a spectrum, not the waveform

**The problem.** An extractor can only pass on what its input holds. Until 2026-11-12 the
autoencoders and PCA were fed the waveform **averaged in 256 blocks**, and averaging a
waveform over a block is a low-pass filter:

```
voice  176400 samples (4 s, 44.1 kHz) ─ 256 blocks of ~689 samples (15.6 ms) ─► nothing above ~32 Hz
                                                                               (pitch: 85–255 Hz, formants: 300–3500 Hz)
EEG    6 × 4096 samples (4 s, 1024 Hz) ─ 256 blocks of 96 samples (94 ms) ───► nothing above ~5 Hz
                                                                               (alpha 8–13 Hz, beta 13–30 Hz: gone)
```

What made a speaker recognisable was removed before any extractor ran.

**The evidence.** `scripts/analysis/thesis_ae_input_reference.py` (scipy + PyTorch, none of
the C++) rebuilds both inputs from the database for the 1974 trials and asks a linear
classifier for the speaker (15 speakers, chance 0.067, 5 folds):

| Input, 256 values | Voice | EEG (pronounced) | Early fusion |
|---|---|---|---|
| Block-averaged waveform (old) | 0.129 | 0.264 | 0.172 |
| Log power spectrum (new) | **0.854** | **0.618** | **0.652** |

**The failure was silent.** Every autoencoder trained, every Phase 01 classifier converged
on its training folds (loss 0.02), and the only symptom was a test EER of 0.48–0.52 for
every autoencoder and PCA profile, which reads as "these extractors are poor". The
Phase 00/01 runs of 2026-11-12 (commits `524da1fd`, `c0607e33`) carry that defect in every
`ae` and `pca` row; their handcrafted rows are unaffected.

**The fix.** `thesis::spectral_input` (`ThesisFeatureExtraction.hpp`): Welch's power
spectrum of the same signal the handcrafted extractor reads.

| Step | Value | Why |
|---|---|---|
| Frame | 512 samples, hop 256 | 257 FFT bins; the 256 above DC are the input |
| Window | symmetric Hann | tapers the frame ends |
| Average | power over all frames | a frame's position (phase) stops mattering |
| `log10`, then min-max per sample | `[0, 1]^256` | the spike encoders read each value as a rate or a spike time |

Samples after the last whole frame (at most 255) are not read. A signal shorter than 512
samples, a non-finite sample, a bin of zero power and a flat spectrum are refused.

**Confusable pair.**

| | Block average (old) | Power spectrum (new) |
|---|---|---|
| Answers | "what was the mean amplitude in this 16 ms?" | "how much energy at this frequency, anywhere in the 4 s?" |
| Same utterance, started 3 ms later | every value changes | (almost) nothing changes |
| Speaker information | below 32 Hz only | the whole band |

**Ground truth.** `thesis_spectrum_parity_gtest` holds `spectral_input` against
`scipy.signal.welch` (worst difference 3e-6 on values in `[0, 1]`) and the PCA of those
vectors against `torch.linalg.svd`; the fixture comes from
`scripts/testing/gen_thesis_spectrum_refs.py`. `ThesisPcaStrategy.
InputKeepsFrequencyAndIgnoresTimeShift` is the regression test for the defect itself.

**LSTM-AE aligned too (2026-11-12, later).** The LSTM-AE (`lstm-ae`, in no generated
profile) read 64 frames of the raw waveform. A profile choosing it would have compared a
different *input*, not a different extractor, and nothing would have said so (silent). It now
reads the same 256 spectral values, as a sequence of `kLstmSteps` = 16 steps of
`kLstmStepWidth` = 16 adjacent bins, low frequency first (`spectrum_to_sequence`). Regression
test: `ThesisLstmAe.ReadsTheSpectrumNotTheWaveform` (two takes differing only in the 100
samples Welch never reads get bit-identical codes).

**What the re-run showed (Phase 00 job 5323830, Phase 01 job 5323900, 2026-11-12).**

| | Block-average input (void) | Spectral input (run of record) |
|---|---|---|
| Phase 00 winner, all three signals | handcrafted `haar/lfcc/c1` | the same |
| Best PCA / best ANN-AE `d_penalized` (voice) | far behind | 1.7230 / 1.7479, still behind the handcrafted winner |
| EER through the DSNN, voice (hc / AE / PCA) | AE and PCA 0.48-0.52 (chance) | 0.3243 / 0.3053 / 0.3523 |
| EER, pronounced EEG | same | 0.2688 / 0.3521 / 0.3757 |
| EER, imagined EEG | same | 0.2504 / 0.2302 / 0.1934 |

No extractor comparison is significant (corrected t + Holm). Two readings follow: the
automated extractors are not worse than the handcrafted one through the classifier, and
`d_penalized` did not predict the order across families (it ranks the handcrafted extractor
far ahead in all three signals; the EER does not). The SNN-AE encodings rank
latency < poisson < direct, the opposite of the noise-floor prediction. The void run is kept
in `results/thesis_blockavg_input_20261112/`.

### 10. No number is typed into the thesis

**Problem.** Chapter 9 held 58 hand-typed results and the abstract 11; after a re-run they
described a run that no longer existed, and nothing failed (silent).

**Mechanism.** `scripts/pipeline/thesis/thesis_build_text_numbers.py` reads the two result
sets and writes `documentation/00-thesis/monography/tables/thesis_numbers.tex`: one key per
number. The text says `\tn{p01.best.eer}` (decimal comma) or `\tnen{...}` (decimal point, for
the English abstract).

| Mistake | What happens |
|---|---|
| A key that does not exist | LaTeX error (loud) |
| A sentence whose claim stopped being true ("no comparison is significant", "the PCA has the lowest EER") | the generator's `CLAIMS` list refuses to write the file (loud) |
| A number typed by hand | nothing (silent) - do not do it |

```bash
python3 scripts/pipeline/thesis/thesis_build_text_numbers.py \
  --phase00-profiles src/experiments/thesis/profiles/phase00 --phase00-results results/thesis/phase00 \
  --phase01-profiles src/experiments/thesis/profiles/phase01 --phase01-results results/thesis/phase01 \
  --ae-input-reference results/thesis/ae_input_reference.csv \
  --out ../../documentation/00-thesis/monography/tables/thesis_numbers.tex   # --list prints the keys
```

It needs the significance files first: `scripts/pipeline/thesis/03_thesis_phase01_significance.sh`
writes one `significance_<family>_*.csv` per declared family (`vsvoice`, `fusion`, `ctrl`,
`dysph`, `cmp`, `text`, `cond`) into `results/thesis/phase01/`.

### 11. A second base: the BIOMEX-DB replication (2026-11-13)

**Problem.** Every thesis number comes from one base of 15 speakers, and the extractor was
chosen on that base (Phase 00). A result that holds there may hold only there. A second,
independent base answers that, but only if nothing is chosen on it: run a search on the new
base and its best cell is a selection again, not a test.

**Design: confirmatory.** The Coretto Phase 00 winners are frozen *before* any BIOMEX-DB
result exists (`results/thesis/biomex/coretto_phase00_winners_frozen.json`: voice and
pronounced EEG both `handcrafted/haar/lfcc/c1`, DTWPT level 4) and the Phase 01 cells the
thesis concludes on run once on BIOMEX-DB:

| Profiles (`profiles/biomex/`) | Cells | Count |
|---|---|---|
| `bx01_{dsnn,ann,cosine}_<src>_<text>_flat_std` | sources voice, eegpron, earlypron, latepron × dep/indep | 24 |
| `bx01_dysph_<src>_<scope>_l<level>` | sources that read the voice × probes/test × 0.25-1.0 | 24 |

No Phase 00, no nested CV, no raw (unstandardized) runs on BIOMEX-DB. It has **no imagined
speech**, so `eegimag` and the imagined fusions do not exist there (`ThesisConfig` refuses
`dataset.source biomex2021` with `eeg_condition imagined`).

**The base, measured on volunteer F001.**

| | Coretto (`coretto2017`) | BIOMEX-DB (`biomex2021`) |
|---|---|---|
| Speakers | 15 | 51 (25 F) |
| Conditions | imagined (EEG only), pronounced (EEG + voice) | pronounced only (EEG + voice) |
| Stimuli | 5 vowels, 6 commands | digits 0-9 (Spanish), 140 per speaker: 14 per digit except 3 (15) and 8 (13), as the paper's Table 1 sequences imply |
| EEG | 6 ch, 1024 Hz, 4 s, 2-40 Hz by the authors | Emotiv Epoc, 14 ch, recorded at 128 Hz, **written at 1024 Hz**; 2 s after the digit marker, 2-40 Hz by the converter |
| Voice | 44.1 kHz, 4 s window | recorded at 16 kHz, **written at 44.1 kHz**; 1 s from the labelled word onset (moved earlier when it would reach the next word) |

**How a pair is made** (`scripts/data/biomex_to_sqlite.py`). The EEG and the voice of a series
are on two clocks that disagree (a series lasts 26.3 s in the EEG, 20.0 s in the WAV), so each
is cut on its own clock and the two are matched by order:

```
EEG MARKER channel   99 89 | 7  9  10 2 ... 3 | 69 | 1 7 10 ... | 69 ...
                           |<-- series 1 ---->|    |<- series 2
voice series 1 .lab        SIETE NUEVE CERO DOS ... TRES
pair j of series k  =  (j-th digit marker of run k, j-th word of WAV k); digits must agree
```

**The voice window, and the speakers who talk faster than it.** Each voice sample is 1 s of
the series WAV starting at the word's labelled onset: the word (p99 0.74 s over F001-F006)
plus the silence after it. Most speakers leave ≥ 1.4 s between digits, but not all:

```
F004 01G04 series 1   UNO at 1.476 s, DOS at 2.002 s  (0.526 s apart)
                      |UNO|..|DOS|.................
1 s from UNO's onset   [1.476 ------------- 2.476)   holds the start of DOS  -> wrong pair content
moved earlier          [1.002 ------- 2.002)          ends where DOS begins; silence before UNO
```

Three ways out, and why the converter takes the third:

| Option | What it costs | Fails how |
|---|---|---|
| shrink the window for everyone to the shortest gap (0.5 s) | cuts every long word (p99 0.74 s) to fit one fast speaker | silently: worse features for all 51 |
| drop the pairs that do not fit | a selection nobody chose; differs per speaker | silently: fewer probes for the fast speakers |
| **move that window earlier, just enough to end at the next onset** | the word sits later in its window (0.47 s for UNO above) | stored per pair (`voice_shift_s`); refused loudly if the moved window would reach the previous word |

The same rule covers a WAV that ends less than 1 s after its last word (F006: 0.996 s). A
labelled word longer than the window is kept from its onset and counted (`voice_cut`; one in
F001-F006, F006's "TRES" labelled 1.676 s). Totals are in the base's `recording` table
(`voice_windows_shifted`, `voice_words_cut`).

**Why the converter filters the EEG, and with the Coretto authors' own filter.** The Emotiv
EEG sits on a DC offset of ~4200 uV that differs per electrode and per headset fit.
Unfiltered, energy and spectrum features would read the fit of the headset and identify
speakers by it — a silent confound that looks like a good EER. The converter applies the
filter the Coretto paper publishes (Pressel Coretto et al. 2017, sec. 2.3), not a filter of
its own, so the two bases differ in recording only:

```
EDF, 128 Hz, ~4200 uV offset
  | 1. subtract each channel's session mean      (the FIR's DC stopband is -53 dB: 4200 uV -> ~9 uV left without this)
  | 2. resample_poly x8 -> 1024 Hz                (the Coretto rate, see below)
  | 3. FIR low-pass, order 372, 40 Hz, Hamming    (Coretto's)
  | 4. FIR high-pass, order 1204, 2 Hz, Hamming   (Coretto's)
  |    both odd-length and symmetric: applied "same" = the 788-sample group delay removed
  v
epoch: 2 s after the marker, >= 788 samples from either end of the session (else refused)
```

`test_the_eeg_filter_is_the_one_the_coretto_paper_gives` pins the orders, -6 dB at 2 and 40
Hz, flatness at 5-30 Hz, the DC rejection, the 10 Hz phase and the offset removal. Both bases
reach the loader in uV and the voice as float PCM in [-1, 1]. The Coretto voice also went
through a 4th-order Butterworth low-pass at 10 kHz; it is not applied to BIOMEX-DB, whose 16
kHz recording holds nothing above 8 kHz, where that filter's loss is ≤ 0.7 dB.

**Rates come from the base.** Until this change the extractors used constants (44100 / 1024
Hz). `ThesisDatasetView` now carries `voice_sample_rate` / `eeg_sample_rate`, set by the
loader (Coretto: its documented rates; BIOMEX-DB: the `recording` table); a view without them
is refused (`require_rate`).

**Why BIOMEX-DB is written at the Coretto rates (decided 2026-11-13).** The problem: the
classifier receives the same *shape* from both bases (one row of 96 numbers), but a column only
means the same thing if it measures the same band of Hz. The frozen winner cuts 0..Nyquist into
2^4 = 16 equal DTWPT sub-bands, so the width of a sub-band is set by the rate:

```
                    rate      Nyquist   sub-band width   power in the 16 sub-bands (%)
Coretto voice       44.1 kHz  22050 Hz  1378 Hz          99.2  0.4  0.1  0.1  0.1  0 ...
BIOMEX voice, 16k   16 kHz     8000 Hz   500 Hz          59.9 27.0  2.1  1.2  2.8  2.5 ... (9 bands hold 99%)
BIOMEX voice, 44.1k 44.1 kHz  22050 Hz  1378 Hz          89.0  5.5  3.2  2.0  0.2  0 ...
Coretto EEG         1024 Hz     512 Hz    32 Hz          97.2  2.8  0 ...
BIOMEX EEG, 128     128 Hz       64 Hz     4 Hz          46.3 32.5  6.8  3.1  2.7 ... (10 bands hold 99%)
BIOMEX EEG, 1024    1024 Hz     512 Hz    32 Hz          96.8  3.2  0 ...
```

(80 random samples per base; BIOMEX from F001-F006.) At its native rates, column "sub-band 0
energy" was 0-1378 Hz of voice on Coretto and 0-500 Hz on BIOMEX-DB, and the EEG columns were
0-32 Hz on one base and 0-4 Hz on the other: the same number for different things. That fails
**silently** — every run completes and reports an EER.

| Option | Same Hz per column | Cost |
|---|---|---|
| keep native rates | no | none; a limitation to report |
| Coretto down to 16 kHz / 128 Hz | yes | redo Phase 00 + 01, new winners, rewrite the numbers |
| **BIOMEX up to 44.1 kHz / 1024 Hz** | yes | a 2 GB base; Coretto results and frozen winners stay valid |

The converter resamples the whole series WAV (`resample_poly` 441/160) and the whole filtered
EEG session (×8) before cutting; the low-pass at the native Nyquist adds no content above 8
kHz / 64 Hz (Coretto itself has 0.1% / 0.0% there). `load_paired` refuses a BIOMEX file at
any other rate, so a native-rate file cannot reach the extractors.

**What still differs, and why it is not a format difference.**

| Difference | Coretto | BIOMEX-DB | Reaches the classifier as |
|---|---|---|---|
| duration | 4 s voice, 4 s EEG | 1 s voice, 2 s EEG | per-column constants (energy, padding share), removed by the per-fold z-score |
| EEG channels | 6 (F3 F4 C3 C4 P3 P4) | 14 (Emotiv) | the same 96 columns: channels are concatenated into one signal before the DTWPT |
| zero padding to 2^n | voice 33%, EEG 25% | voice 33%, EEG 12.5% | per-column constants, removed by the z-score |
| EEG power below 2 Hz | 19.2% | 3.3% | content: sub-band 0 holds it on both. Both now went through the same digital filter (Coretto's order-1204 high-pass, a transition band of a few Hz around 2 Hz), so the gap is the recordings' (Coretto's analog 0.3-35 Hz amplifier band vs the Emotiv's), not the processing; 80 random pronounced epochs per base, same code, each at its own epoch length |
| voice power below 300 Hz | 77.7% | 28.3% | content of the recordings (room, microphone) |

`ThesisBiomexDataset.BothBasesReachTheClassifierInTheSameFormat` runs both shapes through the
frozen winner and requires 96 columns (192 for late fusion), finite, for every source.

**Is the SQLite the base? (`scripts/data/biomex_verify.py`, 2026-11-13).** A converter
cannot see its own mistakes (a wrong EDF offset, a series cut one off, speakers numbered in
another order), so the check uses sources the converter never reads:

| Check | Against | Why it is independent |
|---|---|---|
| every EDF value the converter reads | the session's CSV export (Emotiv TestBench) | another file, written by another exporter; equal to ≤ 2e-4 uV, MARKER exact |
| every series' digits (EEG markers and .lab words) | Table 1 of the paper | the protocol as published, the same 10 + 10 sequences for every volunteer |
| every stored pair | the pair rebuilt from the originals | WAV via `scipy.io.wavfile`, the window rule re-implemented |
| counts | 140 per volunteer; per digit 14, except 3 (15) and 8 (13) | Table 1 |

A mutation test proves it can fail: one swapped voice, one changed label and one swapped
EEG epoch in a copy were all reported, exit status 1. On success it writes
`<db>.verified` (the base's sha256); `gridunesp_deploy.sh` refuses to send a base without
a stamp that matches. Two facts it surfaced: the CSV exports stop at a round row count
(22000 / 43000), so the EDF is the complete recording (markers exist after the CSV ends);
and the EDF records past the CSV are real signal, not padding.

**Which recording is a volunteer's: the strict rule (decided 2026-11-13).** The paper
describes one 01G10 and one 01G04 recording per volunteer. Several folders do not look like
that (`scripts/data/biomex_inventory.py` lists every EDF of every folder):

```
F007/EEG   F007-01G10-…edf   F006-01G04-…edf        <- 01G04 carries F006's name (header and CSV too)
M001/EEG   M001-01G10-… x2 complete, + 02G10, 02G04 <- which 01G10 went with the voice?
M007/EEG   M007-1G10-…edf                           <- not a session name the protocol has
M010/EEG   M10-01G10-…edf                           <- another volunteer id
F008/EEG   F008-01G10-11.07.56.edf  (64 KB, no digit marker) + the real one   <- an aborted start
```

Reading each of these as "obviously" one thing is a guess per file, and a guess that is wrong
pairs one person's EEG with another's voice: the run completes and reports an EER, so the
failure is **silent**. The rule is therefore mechanical and has no per-file exceptions:

| A folder's `EEG/`, per session (01G10, 01G04) | Verdict |
|---|---|
| exactly one recording named `<vid>-<session>-*.edf` holding the complete protocol (10 runs of 10 or of 4 digits) | kept |
| a recording with no digit marker at all (an aborted start) | ignored: it holds no data |
| a recording with some digits but not the whole protocol | volunteer excluded |
| zero or two+ complete recordings under its own name | volunteer excluded |
| 02G10 / 02G04 (a second day, M001, M003, M004) | not part of the protocol, ignored |

`own_recordings` applies it in the converter; `strict_rule` in the verifier is a separate
re-implementation (by file name and marker content), and the verifier refuses unless both reach
the same excluded list, which the base records (`recording.excluded_volunteers`, with reasons).
Four tests build each case (`test_biomex_convert.py`).

One pair is left out by name, not by rule: F017 01G10 series 8 has 9 `.lab` words for 10 markers.
No line covers 126715626-131048750 (100 ns), there is speech there, and the EEG marker and
Table 1 both say 0. Its position is dropped (`UNLABELLED_DIGITS`), applied only if no `.lab`
line overlaps that span (else refused), and the span still counts as spoken for the windows
next to it.

**Cautions (measured, F001).** (1) The two "sessions" are 7 minutes apart on the same day,
not two days. (2) EEG while speaking carries muscle activity: 2 s after a digit marker vs
eyes-open rest, power 30-64 Hz ×2.3 (T7 ×4.4, F7 ×3.8), 1-4 Hz ×11.8 — "pronounced EEG"
identification can be partly jaw and eyes, on both bases.

**And the Coretto base?** Its counts match Tables 3 and 4 of its paper in every cell but one:
S01 /o/ pronounced has 11 trials in `~/database.sqlite`, 10 in the paper (whose column sums to
175, consistent with 10). The 11 are distinct trials from distinct rows of the original matrix,
so the difference is between the paper and the distributed data, not the import. The authors'
Google Drive folder now requires a sign-in (`gdown` fails), so the blobs are compared with the
only other copy: an EEG-only re-host on Zenodo (record 19502780, uploaded for MOABB by a third
party, files renamed `sub-NN_eeg.mat`). `scripts/data/coretto_verify_mirror.py` checks every
row of it against the trial with that `original_row`: six channels exactly, modality, stimulus,
blink, and every row present once. Agreement there means two independent derivations agree, not
that either is certified; the audio has no second copy and stays unverified.

```bash
python3 scripts/data/biomex_download.py --dest ~/biomex            # 8.1 GB, keeps EEG + Voice, SHA-256 checked
python3 scripts/data/biomex_inventory.py --src ~/biomex/extracted  # every EDF of every folder; the irregular ones
python3 scripts/data/biomex_to_sqlite.py --src ~/biomex/extracted --out ~/biomex.sqlite
python3 scripts/data/biomex_verify.py --src ~/biomex/extracted --db ~/biomex.sqlite  # stamp
THESIS_PHASE=biomex sbatch --array=0-47 scripts/pipeline/thesis/thesis_gridunesp.sbatch
```

---

## Theoretical Background

### Problem: Voice-Based Authentication Under Dysphonia

Conventional speaker-verification systems assume a clean, periodic voice signal. In severe laryngeal dysphonia, the phonation mechanism is impaired: aperiodic vibration, breathiness, and noise dominate the signal, obscuring the formant structure that encodes speaker identity.

Imagined (covert) speech is proposed here as a complementary signal: the speaker mentally rehearses an utterance without producing motor output. The perisylvian language network (Broca + Wernicke areas) activates and EEG captures neural correlates that do not depend on laryngeal phonation, so imagined-speech EEG remains available even when the voice signal is degraded.

### Paraconsistent Feature Selection (EPC/α/β)

Before training any classifier, the pipeline ranks all feature-strategy×modality combinations using the paraconsistent quality metric. This avoids expensive hyperparameter sweeps.

Features are scaled **per dimension to [0,1] across all samples** before scoring
(audit M-1), so every descriptor is commensurable and bounded. Then:

$$\alpha = \min_c \left( \frac{1}{F} \sum_{d=1}^{F} \big(1 - (\max_{c,d} - \min_{c,d})\big) \right) \quad \text{(intraclass similarity)}$$

where $\max_{c,d}, \min_{c,d}$ are the per-dimension extrema within class $c$ over its $M$ feature vectors.

$$\beta = \frac{R}{N\,(N-1)\,M\,F} \quad \text{(interclass overlap)}$$

where $R$ counts — over all ordered class pairs — how many feature-vector components fall inside another class's per-dimension $[\min, \max]$ range; $N$ classes, $M$ vectors per class, $F$ dimensions.

Map to paraconsistent plane:

$$G_1 = \alpha - \beta \qquad G_2 = \alpha + \beta - 1$$

$$D_{\text{truth}} = \sqrt{(G_1 - 1)^2 + G_2^2}$$

Smaller $D_{\text{truth}}$ → features closer to the "Truth" corner → better speaker separability before any classifier sees them. The combination with the minimum $D_{\text{truth}}$ is passed to Stage E4.

See [Paraconsistent Feature Engineering](../Core/Paraconsistent.md) for the full derivation and API.

### Handcrafted Feature Extraction (DTWPT-based)

The Discrete-Time Wavelet Packet Transform decomposes the signal into a full binary tree of sub-bands. For a signal of length N with a J-level DWPT:

$$x[n] = \sum_{j,k} c_{j,k} \psi_{j,k}[n]$$

The **mother wavelet** $\psi$ is a Phase-00 sweep axis (`handcrafted.wavelet`): any tag with coefficient traits in `include/wavelet/Types.hpp` — `haar` or `daubN` for even N in [4, 46] (23 in total). Shorter filters (e.g. Haar) localise sharp transients; longer Daubechies filters give smoother, more selective sub-bands. The best wavelet per signal is chosen by the paraconsistent ranking.

At each leaf node (sub-band), the following descriptors are computed:

| Descriptor | Formula | Sensitivity |
|---|---|---|
| Sub-band energy | $E_k = \sum_n c_k[n]^2$ | Overall power per band |
| ZCR | $\frac{1}{N}\sum_n \mathbb{1}[\text{sign}(x[n]) \neq \text{sign}(x[n-1])]$ | Voicing / noisiness |
| Entropy | $-\sum_k p_k \log p_k$ | Spectral spread |
| Teager–Kaiser | $\Psi[x(n)] = x^2(n) - x(n+1)x(n-1)$ | Amplitude modulation |
| Jitter | $\frac{\overline{|T_i - T_{i+1}|}}{\bar{T}}$ | Pitch period irregularity |
| Shimmer | $\frac{\overline{|A_i - A_{i+1}|}}{\bar{A}}$ | Amplitude irregularity |

Frequency scales evaluated: **BARK**, **MEL**, **LFCC** (see [LFCC](../Concepts/LFCC.md)).

> **The `scale` axis is voice-only (fixme.md D6).** Bark and Mel are **cochlear** scales — they model the frequency resolution of human *hearing*. There is no physiological basis for applying them to EEG, which is not sound, so `ThesisConfig::validate()` rejects `scale != lfcc` when `modality=eeg`.
>
> They were also provably inert there, which is what exposed the problem. `group_by_scale()` normalizes the perceptual curve by the signal's **own Nyquist**. Bark spans ~24 Barks over the audible range and `n_bands=24`, so for voice the factor is ~0.97 — a no-op, the bin *is* the Bark number, and the curve's compression at high frequency genuinely merges sub-bands (bark→9 groups, mel→11, vs lfcc's 16). For EEG (Nyquist 512 Hz) the factor is ~4.96: the curve is stretched 5× to fill 24 bins. Bark/Mel are ~linear over that range, so the mapping becomes injective — each sub-band lands in its own bin and the grouping degenerates to *exactly* lfcc's one-group-per-sub-band. "Bark" for EEG was never Bark; it was a linearly rescaled pseudo-scale identical to linear. Verified on the stored results: all three scales gave bit-identical `D_truth` in **46/46** EEG wavelet×category groups (the only apparent exceptions, `daub32`, differed by ~1e-6 — float noise from that profile's individual re-run).
>
> Consequence worth knowing: because the three tied exactly, the EEG Phase-00 winner was decided by the ranking's **sort tie-break** and came out labelled `bark`. Reporting "the Bark scale won for EEG" would have been false — Bark did nothing, and the winning vector is literally the linear one. `01_thesis_phase00_rank.py` now detects and reports exact ties so an arbitrary tie-break can't be mistaken for a result. Guarded by `ThesisEegScaleAxis.BarkAndMelAreRejectedForEeg`.

**Category 1 vs Category 2** (`handcrafted.cepstral`). With `cepstral=false` the per-band energies are the features (Category 1: linear/Mel/Bark-band energies). With `cepstral=true` a **log + DCT-II** is applied over the band energies, yielding cepstral coefficients — **LFCC / MFCC / BFCC** for `scale = lfcc / mel / bark` (Category 2). The cepstral coefficients replace the raw energy descriptor; other descriptors (ZCR, entropy, …) are still appended per band.

### Learned Feature Extraction (Autoencoders)

> **Scope note.** The **thesis** Phase 00 compares three feature-extraction routes through the paraconsistent ranking: **handcrafted**, **SNN-AE** (spiking autoencoder, `ProtocolSpikingAutoencoder`), and **ANN-AE** (non-spiking dense autoencoder, `ProtocolAutoencoder`). Both AE families are wired into the Experiment05 executable and shipped as Phase 00 profiles. The **LSTM-AE** remains in the code (built for the Meeting01 congress paper) but no thesis profile uses it.

**SNN-AE (`ProtocolSpikingAutoencoder`, implemented)**: spiking autoencoder. Encoder `Linear → LIF`, decoder `Linear → LIF-integrator`. Its input is the signal's 256-bin log power spectrum, **min-max normalized to `[0,1]`** (`spectral_input`, [Protocol revision §9](#9-what-the-autoencoders-and-pca-read-a-spectrum-not-the-waveform); until 2026-11-12 it was the waveform averaged in 256 blocks, which held no speaker information).

*Temporal coding (required for a meaningful SNN).* A LIF autoencoder only carries information through spikes, so each normalized sample is expanded into `time_steps` spike frames (16 in every Phase 00 profile) using `autoencoder.encoding`:
- `poisson` — each entry spikes with Bernoulli probability = its value (rate code);
- `latency` — stronger inputs fire earlier (`t_spike = round((1−v)(T−1))`, matches the Experiment04 encoder);
- `direct` — analog pass-through, no spikes (the non-temporal SNN-AE baseline; takes no `time_steps`). Any other value throws.

The AE is trained (batched, MSE reconstruction) to reconstruct the spike frames. At readout the membrane state is **reset once per sample** and then **integrates across the T frames** (no reset between frames) — the temporal integration that lets weak, sub-threshold per-step current accumulate into spikes. The **per-sample feature is the mean latent over the T frames**.

Two knobs make this work and both are profile-configurable:
- `autoencoder.time_steps` (16 in the profiles; required for `poisson`/`latency`, refused for `direct`) — integration window.
- `autoencoder.voltage_threshold` (**0.2** in the profiles, vs the LIF layer's own 1.0; required for `snn-ae`) — the encoder LIF threshold. Normalized spike frames are low-amplitude, so at `V_th=1` no encoder neuron fires and the latent collapses to all-zeros. That collapse (plus the old single-analog-vector, one-forward presentation) is why the raw-vector SNN-AE ranked poorly. (Audit m-2: this note previously glossed `α≈0.5` as "coin-flip separability" — that reading came from the inverted convention. Under the correct one, `α≈0.5` is mid-range, and the best-ranked configs of the whole grid sit at `α≈0.64–0.67`; see the corrected table below.)

Spike frames are seeded from `experiment.seed` for reproducibility. Guarded by `ThesisSnnAe.*` tests (latent non-degenerate + varies across samples; encoding changes the feature).

> **Dead-latent guard: encoder firing-rate regularization (2026-07-16).** A collapsed encoder (constant output regardless of input) lands exactly on the paraconsistent *Ambiguity* vertex (`α=β=1`) and used to score deceptively well under `D_truth` alone — see [Core/Paraconsistent.md](../Core/Paraconsistent.md) for the `D_penalized` metric that closed that ranking exploit, and [Spike-Rate-Regularization](../Concepts/Spike-Rate-Regularization.md) for why a starved LIF encoder collapses in the first place (root cause: at He/Kaiming init the single spiking stage of `linear:16:leaky, linear:8:identity` fires at only ~8.3%, and the `identity` second stage adds no further nonlinearity, so a starved first stage starves the whole latent). `feature_extraction.autoencoder.{firing_rate_reg_lambda, firing_rate_min, firing_rate_max}` push each encoder `Lif` layer's mean firing rate into `[firing_rate_min, firing_rate_max]`, mirroring the DSNN classifier's mechanism (same math, see below) but implemented separately in `ProtocolSpikingAutoencoder::backward` — the AE's encoder is a generically-built `Sequential` (from `encoder_layer_spec`), not the classifier's named layers, so the Lif layers are located once at construction via `dynamic_cast`, and the regularization gradient is injected by a manual reversed-order backward loop rather than `Sequential::backward`. All 18 real `snn-ae` Phase 00 profiles (and their 18 `smoke` mirrors) now set `firing_rate_reg_lambda=0.5`, `firing_rate_min=0.10`, `firing_rate_max=0.80` — `0.10` sits above the measured native init rate (8.3%) so the penalty actually engages; `0.5` was validated on `direct` at the profile's real learning rate (baseline `α=0.5` → `α=0.375` with regularization on). `firing_rate_reg_lambda` is required for every `snn-ae` (0 = off); the band keys are required exactly when it is > 0. Guarded by `AutoencoderRunnerRedesignTest.ProtocolSnnFiringRateRegularization*` (inject-when-enabled / inert-when-zero).
>
> The 300 stored Phase 00 results predate this change and do not yet reflect it — re-running the 18 `snn-ae` profiles to regenerate results under the new setting is a real cost (the `poisson`/`latency` encodings, T=16, took over 2h for a single run in testing) and has not been done.

**Encoding sweep (thesis comparison).** Phase 00 ships all three SNN-AE encodings — `poisson`, `latency`, `direct` — × the 3 compact sizes × 2 sources = 18 profiles (`p00_ae_snn_<encoding>_<size>_<source>.json`), so the thesis can directly compare rate coding, first-spike-time coding, and the (expected-degenerate) non-temporal baseline through the same paraconsistent ranking.

All 18 profile names (`profiles/phase00/`):

| encoding | tiny | small | base |
|---|---|---|---|
| `poisson` | `p00_ae_snn_poisson_tiny_eeg.json`, `..._voice.json` | `p00_ae_snn_poisson_small_eeg.json`, `..._voice.json` | `p00_ae_snn_poisson_base_eeg.json`, `..._voice.json` |
| `latency` | `p00_ae_snn_latency_tiny_eeg.json`, `..._voice.json` | `p00_ae_snn_latency_small_eeg.json`, `..._voice.json` | `p00_ae_snn_latency_base_eeg.json`, `..._voice.json` |
| `direct`  | `p00_ae_snn_direct_tiny_eeg.json`, `..._voice.json`  | `p00_ae_snn_direct_small_eeg.json`, `..._voice.json`  | `p00_ae_snn_direct_base_eeg.json`, `..._voice.json`  |

Each profile is identical except `feature_extraction.autoencoder.{encoding, time_steps, voltage_threshold}` and `encoder_layer_spec`/`decoder_layer_spec` widths (per size, see table below). `direct` profiles use `time_steps: 1` (ignored by the encoder anyway — a single analog frame) and `voltage_threshold: 1.0` (the LIF default — appropriate since `direct` is meant to reproduce the *un-encoded* baseline, not a tuned spiking regime); `poisson`/`latency` use `time_steps: 16`, `voltage_threshold: 0.2`.

> ### ⚠️ Encoding comparison: withdrawn pending re-run (fixme.md D2/F4)
>
> A table used to sit here reporting measured `α`/`D_truth` per SNN-AE encoding (`direct` best,
> `poisson` last) over the Phase 00 grid. **It has been removed rather than corrected**, because
> every one of its numbers is unsupported and none can be recomputed:
>
> 1. **Produced under the `snn_lr_scale` bug (D3).** Every AE weight trained at an effective
>    lr of 1e-4 while the profiles declared 1e-3. The numbers describe a run nobody configured.
> 2. **Its ranks are arithmetically impossible now.** It said "rank (of 150, eeg)"; after D6
>    retired the degenerate EEG bark/mel axis, EEG has **58** combinations (46 handcrafted + 12 AE).
> 3. **The source data is gone.** All Phase 00 results were deleted (see `results/thesis/phase00/README.md`);
>    the re-run is postponed, so the table cannot be regenerated.
>
> Correcting the figures was not an option — there are no figures to correct. Restating them with a
> caveat would have kept three-decimal precision on numbers with no run behind them, which is how
> the *previous* revision of this table (a 20-epoch/2-fold preliminary run, reported as if final and
> with the direction backwards) survived long enough to reach the thesis.
>
> **What still stands** is the mechanism, which is theory rather than measurement, and which
> predicts the ordering independently of any run:
>
> - `α` is built from the intraclass **min–max range**, so it is maximally outlier-sensitive: a few
>   extreme samples per class widen the range to the whole normalised interval and drive `α→0`.
> - Each encoding injects a different, *quantifiable* amount of noise **before any learning happens**.
>   `poisson` redraws an independent Bernoulli per time step, so the mean over `T=16` frames carries
>   variance `v(1-v)/T` — at `v=0.5` that is a standard deviation of **≈0.125, i.e. ~12.5% of the
>   feature's full [0,1] range**, added per sample regardless of how well the encoder trained.
>   `latency` is deterministic: the same `v` always yields the same spike time, and its quantisation
>   error is at most `½·1/(T−1) ≈ 0.033`. `direct` injects none.
> - That ladder (none / small-deterministic / large-random) predicts the observed ordering, and
>   predicts it as an artifact of the **metric plus the encoder's noise floor**, not as evidence that
>   temporal codes carry less speaker information.
>
> So the earlier framing — "a negative result for temporal coding" — **is not established**, and is
> not merely uncertain: `α` cannot separate "the encoder failed to learn" from "this encoding has an
> irreducible statistical floor at T=16". Since the floor scales as `1/T`, the claim is testable by
> re-running `poisson` at larger `T` (T=64 shrinks the variance 4×). That test has **not** been run.
> See fixme.md **D2** for the decision and the derivation.

**ANN-AE (`ProtocolAutoencoder`, implemented)**: non-spiking dense autoencoder — same flat 256-value spectral input and 2:1 compression, ReLU activations. Serves as the non-spiking baseline against SNN-AE.

**LSTM-AE (legacy, Meeting01 paper — not in the thesis Phase 00 grid)**: sequence-to-sequence autoencoder on the same 256-value spectral input as the others, read as 16 steps of 16 bins. Encoder LSTM processes the steps, final hidden state = latent vector; decoder LSTM reconstructs the frame sequence. Trained with MSE + BPTT. See [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md).

> **Windowing + batching (AE-on-EEG fix, superseded 2026-11-12 by the spectral input above).** The raw signal used to be *framed* into at most 64 windows of `frame_len` samples each → AE input `(T_frames, frame_len)`. This replaced the earlier `input_size=1, seq_len=24576` wiring, which fed the whole flattened multi-channel EEG as one length-24576 sequence — both semantically wrong and far too long to unroll (and it crashed once the `LSTMAutoencoder` met the trainer's batched 3-D tensor). `LSTMAutoencoder` now handles both 2-D `(T,D)` and 3-D `(B,T,D)` inputs (`LSTMLayer` already did the batched BPTT; the projections/last-step/replicate were made batch-aware). Verified against snnTorch/PyTorch — see [Ground-Truth and Smoke Testing](../Guides/Ground-Truth-and-Smoke-Testing.md).

**Compact-AE capacity sweep (low-power design).** Phase 00 ships two AE families (SNN-AE, ANN-AE) × three compact sizes per signal, all single-layer (shallow, edge-friendly) with a 2:1 hidden→latent compression ratio:

| size | hidden | latent (= feature-vector dim) |
|---|---|---|
| `tiny`  | 16 | 8  |
| `small` | 32 | 16 |
| `base`  | 64 | 32 |

The latent layer *is* the feature vector, so a smaller bottleneck yields a smaller vector — cheaper downstream classifier and often better generalisation. These choices follow the lightweight-autoencoder-for-edge literature (aggressive bottleneck, ~2:1 compression, shallow stack). The wired families are `snn-ae` and `ann-ae` (each swept over the three sizes); `lstm-ae` is accepted by the config but no thesis profile ships it. See [Autoencoders](../Concepts/Autoencoders.md) and [Memory-Constrained Design](../Guides/LSTM-Performance.md).

Autoencoder training in Experiment05 is unsupervised (no speaker labels), and latent vectors feed paraconsistent ranking and downstream classifier.

### Authentication: Deep SNN (DSNN) and its controls

The thesis classifier is the **DSNN**. Two controls run on the same features and folds
(see [Protocol revision §3](#3-controls-what-does-spiking-buy-what-does-a-network-buy)):
**`ann`** (`ThesisAnnClassifier.hpp`), the DSNN's graph with ReLU for LIF and the DSNN's
initial weights, and **`cosine`**, no network. The type `rnn` (`SimpleResNet`, a different and
larger graph) was removed on 2026-11-12.

**DSNN**: temporal deep **residual** spiking classifier (audit M-4). Each hidden block (`Linear → (tdBN) → LIF`, all `hidden_dim` wide) is wrapped in an identity skip connection `h_out = block(h) + h`; `residual:D` in the layer spec sets the number of such blocks. Skips are parameter-free (no added serialization state) and sit only around the equal-width hidden blocks — the input stage (`input_dim → hidden_dim`) and output stage carry none. This matches the deep-residual-SNN design that tdBN was introduced to train (Zheng et al., AAAI 2021). The static feature vector is **rate-encoded by constant-current injection over `kSnnTimeSteps` (default 16) time steps**; `LifBPTT` neurons integrate the resulting current over time (time-major `(T*B, F)` layout) through a `Linear → LIF → (Linear → LIF)* → Linear` stack. The readout is the **mean firing rate (spike-count / T)** of the final layer, yielding class logits; gradients flow via full backpropagation-through-time with surrogate spike derivatives. This is a genuine temporal SNN, not a single-step thresholding net. See [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) and [Layers](../Core/Layers.md).

### Regularization & Normalization

Three techniques guard generalisation / trainability on the small dysphonic-speaker dataset, each declared per profile (no key has a default since 2026-11-11):

- **Decoupled L2 weight decay** (`training.weight_decay`, AdamW): applies to **both** the RNN and DSNN classifiers. Shrinks only 2-D weight matrices; biases and the SNN biophysical scalars (R, C, V_th) are excluded so `τ = R·C` and the firing threshold are never decayed. See [Optimizers](../Core/Optimizers.md#decoupled-weight-decay-adamw--sgdw).
- **Firing-rate regularization** (`training.firing_rate_reg_lambda` for the DSNN classifier; `feature_extraction.autoencoder.firing_rate_reg_lambda` for the SNN-AE encoder): pushes each spiking layer's mean firing rate into the band `[firing_rate_min, firing_rate_max]` (both bounds required when the penalty is on), preventing **dead neurons** (rate → 0, the surrogate gradient vanishes) and **bursting neurons** (rate → 1, selectivity lost). The penalty $\lambda\sum_\text{layers}\big(\max(0, r_\text{min}-r)^2 + \max(0, r-r_\text{max})^2\big)$ is differentiated to `2λ(r − clamp(r, r_min, r_max))/n` and injected into the incoming gradient at each LIF spike output during backward — the same math as `SpikeCountLossImpl`, but two independent gradient-injection implementations: `ThesisDsnnClassifier::add_firing_rate_grad` (classifier, named layers) and `ProtocolSpikingAutoencoder`'s `backward_with_firing_rate_reg` (AE encoder, `Sequential`-based, Lif layers located via `dynamic_cast`; see the SNN-AE section above). Inert when `λ = 0` (the RNN classifier and the ANN-AE have no spiking layers to regularize either way).
- **Threshold-Dependent Batch Normalization** (`training.batch_normalization = "threshold-dependent"`, DSNN-only): inserts a tdBN layer after each `Linear` and before each `LifBPTT` (`fc_in → tdBN → lif_in → (hidden_fc → tdBN → hidden_lif)* → fc_out`). It normalizes the pre-spike current over batch+time and rescales it to `N(0,(α·V_th)²)` (α = `training.tdbn_alpha`, required when tdBN is on), stabilizing deep-SNN training and guarding against the No-Spike Problem. See [Threshold-Dependent Batch Normalization](../Concepts/Threshold-Dependent-Batch-Normalization.md). Inert for the RNN classifier.

**Input feature standardization** (`training.standardize_features`, required whenever the classifier runs): before the classifier, each feature dimension is z-scored. The mean/std are fit on the **training rows of each fold only** and applied to train and test, so test statistics never leak into the scaler (`fit_scaler`/`apply_scaler` in `ThesisClassifiers.cpp`). This is the CMVN-style input normalization + leakage guard described in [Data Normalisation](../Concepts/Data-Normalisation.md); unlike the three techniques above it is a preprocessing step, not a classifier-only regularizer. Constant dimensions (std ≈ 0) map to 0.

### Seen vs unseen test phrases (`classifier.text_mode`)

Both modes train on the same 5 phrases; `dependent` tests on those 5 (seen), `independent` on
5 others (unseen). See [Protocol revision §2](#2-seen-and-unseen-phrases-on-the-same-training-set).

### Nested 5-Fold Cross-Validation

To avoid optimistic bias from hyperparameter tuning on the test fold, a nested k-fold scheme is used when `training.nested_cv=true`:

- **Outer loop** (5 folds): hold out one fold as test set; report final metrics here
- **Inner loop** (5 folds within training set): model selection by inner validation accuracy

When `training.nested_cv=false`, flat grouped k-fold is used.

See [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md).

---

## Dataset

**10.1117/12.2255697** — EEG imagined speech (public):
- 15 Spanish-speaking subjects
- Utterances: vowels (`/a/ /e/ /i/ /o/ /u/`) + 6 directional commands (`arriba/abajo/adelante/atras/derecha/izquierda`)
- Two speech conditions: **pronounced** (audio + EEG recorded simultaneously) and **imagined** (EEG only — no audio was recorded in the imagined condition)
- Audio: 44100 Hz, single channel (present only for pronounced trials)
- EEG: 1024 Hz, 6 channels (F3, F4, C3, C4, P3, P4), 10-20 system; each 4 s trial = 4096 samples/channel
- Profiles: `dataset.source "coretto2017"`, `dataset.root "~/database.sqlite"`

**BIOMEX-DB** (Moreno-Rodriguez et al., IEEE Access 2021; data 10.17632/s7chktmb6x.1, CC BY 4.0)
— the confirmatory replication of the pronounced-speech results (section 11 above):
`dataset.source "biomex2021"`, `dataset.root "~/biomex.sqlite"` (the paired SQLite written by
`scripts/data/biomex_to_sqlite.py`; table `recording` declares its format, rates, channels and
lengths, and the loader refuses a file whose blobs do not match them).

> The pipeline's `modality` field (`voice` | `eeg` | `fused`) selects which signal(s) feed feature extraction — it is not a dataset-level modality. Fusion is not a third recorded signal; it combines the two existing signals, and *when* the combination happens is a second axis, `fusion_mode` (required for `modality=fused`, refused otherwise):
>
> | `fusion_mode` | Where combination happens | Extractor runs | Result vector |
> |---|---|---|---|
> | `early` | Raw voice+EEG samples concatenated **before** extraction | one pass over the joined signal | features of the joined signal |
> | `late` | Voice and EEG extracted **independently**, feature vectors concatenated **after** | two passes (one per signal) | `[voice-features ‖ eeg-features]` |
>
> Early fusion lets the extractor model cross-modal interactions but forces one sample rate onto both halves (the voice rate, since audio dominates the sample count); late fusion keeps each signal at its native rate and its own extractor, at the cost of never seeing the two jointly. Both are worth comparing as an experimental axis. Implemented in `ThesisFeatureExtraction.cpp::extract_features` (audit C12).

See [Data Loaders](../Core/DataLoaders.md) for the `ThesisDataset` loader API and file layout.

---

## Implementation

### Module plan

```
src/experiments/thesis/
├── thesis.cpp              CLI entry point
├── lib/
│   ├── include/
│   │   ├── ThesisConfig.hpp         profile JSON parser
│   │   ├── ThesisDataset.hpp        EEG + voice loader (wraps 10.1117 loaders)
│   │   ├── ThesisFeatureExtraction.hpp  handcrafted + autoencoder pipeline
│   │   ├── ThesisParaconsistent.hpp paraconsistent ranking step
│   │   ├── ThesisClassifiers.hpp    RNN / DSNN authentication
│   │   └── ThesisOutput.hpp         CSV, JSON, DAT writers
│   └── src/
│       ├── ThesisConfig.cpp
│       ├── ThesisDataset.cpp
│       ├── ThesisFeatureExtraction.cpp
│       ├── ThesisParaconsistent.cpp
│       ├── ThesisClassifiers.cpp
│       └── ThesisOutput.cpp
├── profiles/
│   ├── debug.json                              (RNN quick smoke)
│   ├── phase00/  Phase 00 — feature construction + paraconsistent ranking (classifier.enabled=false)
│   │   ├── p00_hc_<wavelet>_<scale>_<cat>_<source>.json  wavelet(23) ×
│   │   │       cat ∈ {c1=energy, c2=cepstral LFCC/MFCC/BFCC} ×
│   │   │       scale: voice = bark/mel/lfcc (138); eeg = lfcc only (46)  = 184
│   │   │       (eeg drops bark/mel: cochlear scales, no basis for EEG — see D6 note below)
│   │   ├── p00_ae_ann_<size>_<source>.json           ANN-AE, size ∈ {tiny,small,base}
│   │   │                                              (latent 8/16/32, 2:1 hidden) × source(2) = 6
│   │   ├── p00_ae_snn_<encoding>_<size>_<source>.json SNN-AE, encoding ∈ {poisson,latency,direct}
│   │   │                                            × size ∈ {tiny,small,base} × source(2) = 18
│   │   │                                            (6 ann + 18 snn = 24 AE)
│   │   └── p00_pca_k<k>_<source>.json               PCA, k ∈ {8,16,32} × source(2) = 6
│   │                                                → 184 + 24 + 6 = 214 total
│   └── phase01/  Phase 01 — DSNN authentication, best combo only (classifier.enabled=true)
│       └── p01_dsnn_<source>_<text>_<cv>_<std>.json   source(4) × text(dep/indep) ×
│           cv(nested/flat) × std ∈ {std,raw} (standardize_features ablation) = 32
│           (feature_extraction = placeholder; set the Phase-00 winner before running)
  └── tests/
  ├── thesis_profile_audit_gtest.cpp        profile schema and validate() audit (all official profiles)
  ├── thesis_feature_extraction_gtest.cpp   descriptor functions + extract_handcrafted
  └── thesis_classifiers_gtest.cpp          batch evaluate() + aggregate stats + DSNN smoke path
```

### Profile schema — every key declared, no defaults (2026-11-11)

**The problem this solves.** Until 2026-11-11 a profile could omit a key and the config struct
filled it in. The worst case: `autoencoder.model` defaulted to `lstm-ae`, so a profile with
`strategy = "autoencoder"` and no `model` ran the LSTM-AE — an extractor the thesis had dropped —
and its result looked like any other. A census of the 495 profiles then in the tree found 49
that relied on the implicit `ae_loss_type = "mse"`, and 1 000+ keys the run never read (e.g.
`early_stop_patience` in runs with no classifier) that looked like settings but changed nothing.

**The rule.** A key is either *read by this run* or *not*. Read ⇒ the profile must declare it;
not read ⇒ the profile must not contain it. Unknown keys and wrong JSON types (`"epochs": 50.0`,
`"seed": -1`) are refused. There is no default anywhere: `ThesisConfig` holds every value in a
`thesis::Required<T>`, and reading one that was never declared throws (in code-built configs
too — tests use `tests/ThesisTestConfig.hpp`).

**Which keys a run reads** (one table, `for_each_field` in `ThesisConfig.cpp`, feeds the parser,
the declaration check and the summary's `config` block):

| Block | Key | Read when |
|---|---|---|
| `experiment` | `run_tag`, `seed`, `repeats` | always |
| | `seed_deterministic` | `repeats > 1` |
| `dataset` | `root`, `results_dir`, `modality`, `max_samples` | always |
| | `fusion_mode` | `modality = "fused"` |
| `feature_extraction` | `strategy` | always: `handcrafted` \| `autoencoder` \| `pca` |
| `…handcrafted` | `transform`, `scale`, `wavelet`, `cepstral`, `descriptors`, `dtwpt_level` | strategy handcrafted |
| `…autoencoder` | `model`, `encoder_layer_spec`, `ae_loss_type` | strategy autoencoder |
| | `decoder_layer_spec` | model `snn-ae` or `ann-ae` (not `lstm-ae`) |
| | `encoding`, `voltage_threshold`, `firing_rate_reg_lambda` | model `snn-ae` |
| | `time_steps` | `snn-ae` with encoding ≠ `direct` |
| | `firing_rate_min`, `firing_rate_max` | `snn-ae` with `firing_rate_reg_lambda > 0` |
| `…pca` | `n_components` (1–256, and ≤ samples − 1 at fit) | strategy pca |
| `paraconsistent` | `enabled` | always |
| `classifier` | `enabled` | always |
| | `type`, `layer_spec`, `text_mode` | classifier enabled |
| `training` | `epochs`, `learning_rate`, `optimizer_type`, `gradient_clip_norm`, `samples_per_batch` | strategy autoencoder, or classifier enabled |
| | `optimizer_momentum` | optimizer `sgd` |
| | `k_folds`, `nested_cv`, `standardize_features`, `weight_decay` | classifier enabled |
| | `early_stop_patience` | classifier enabled and `nested_cv` |
| | `firing_rate_reg_lambda`, `batch_normalization` | classifier `dsnn` |
| | `firing_rate_min`, `firing_rate_max` | `dsnn` with `firing_rate_reg_lambda > 0` |
| | `tdbn_alpha` | `batch_normalization = "threshold-dependent"` |

So a handcrafted Phase 00 run has **no `training` block at all** (nothing trains), and a PCA
run has none either. Three real profiles, one per strategy:

```jsonc
// phase00/p00_hc_haar_bark_c1_voice.json — handcrafted, no training block
{ "experiment": {"run_tag": "e05_p00_hc_haar_bark_c1_voice", "seed": 42, "repeats": 3,
                 "seed_deterministic": false},
  "dataset": {"root": "~/database.sqlite", "results_dir": "results/thesis/phase00",
              "modality": "voice", "max_samples": 0},
  "feature_extraction": {"strategy": "handcrafted",
    "handcrafted": {"transform": "dtwpt", "scale": "bark", "wavelet": "haar", "cepstral": false,
                    "descriptors": ["energy","zcr","entropy","teager","jitter","shimmer"],
                    "dtwpt_level": 4}},
  "paraconsistent": {"enabled": true},
  "classifier": {"enabled": false} }

// phase00/p00_ae_snn_latency_small_eeg.json — the AE trains, so training is read
  "feature_extraction": {"strategy": "autoencoder", "autoencoder": {
      "model": "snn-ae", "encoder_layer_spec": ["linear:32:leaky", "linear:16:identity"],
      "decoder_layer_spec": ["linear:32:leaky", "linear:output:identity"],
      "encoding": "latency", "time_steps": 16, "voltage_threshold": 0.2,
      "firing_rate_reg_lambda": 0.5, "firing_rate_min": 0.1, "firing_rate_max": 0.8,
      "ae_loss_type": "mse"}},
  "training": {"epochs": 100, "learning_rate": 0.001, "samples_per_batch": 32,
               "optimizer_type": "adam", "gradient_clip_norm": 0.0}

// phase00/p00_pca_k16_voice.json — closed form, no training
  "feature_extraction": {"strategy": "pca", "pca": {"n_components": 16}}
```

**Same mistake, before and after:**

| Profile slip | Before 2026-11-11 | Now |
|---|---|---|
| `strategy: "autoencoder"`, `model` omitted | ran `lstm-ae` silently | refused: `feature_extraction.autoencoder.model` is required |
| `learning_rate` omitted | resolved from the optimizer (adam 1e-3, …) | refused |
| `early_stop_patience` in a run with no classifier | ignored silently | refused: the run never reads it |
| `"epochs": 50.5` | truncated to 50 | refused: wrong JSON type |

Every failure is **loud** (the run stops at parse time with the key's path and the remedy);
there is no silent path left. The run summary records `config` = exactly the declared keys,
so the value that ran is the value written in the profile. Contract tests:
`thesis_strict_config_gtest.cpp`; every shipped profile: `thesis_profile_audit_gtest`.

**paraconsistentGA** embeds this config (`GaConfig::embedding()`): the genome sets
`encoder_layer_spec`, `decoder_layer_spec`, `encoding`, `time_steps`, `voltage_threshold`, so
GA profiles must *not* declare them; the GA validates (instead of forcing) `strategy =
autoencoder`, `classifier.enabled = false`, `paraconsistent.enabled = true`, and probes each
encoding choice it may produce against the same rules.

### Progress bars (three levels)

The experiment shows three concurrent progress bars during a run:

| Bar | Created by | Tracks |
|---|---|---|
| `Feature extraction` | `extract_features()` | samples processed (OpenMP parallel) |
| `Fold N/K \| <label>` | `ProgressCallback` (per outer fold) | epoch + batch within that fold |
| `Thesis \| <run_tag>` | `thesis.cpp` main | outer folds completed across all feature sets |

All bars are rendered by `nn::progress::ProgressManager` (background thread, ANSI escape codes). The global bar is completed and `ProgressManager::shutdown()` called before any output is written.

### Pipeline flow

```
10.1117/12.2255697 dataset
  │
  ├── Pronounced speech (44100 Hz, single channel)
  │     └── preprocessing (normalization, pre-emphasis, windowing)
  │           ├── Handcrafted: DTWPT + ZCR + entropy + Teager + jitter/shimmer
  │           └── Learned: autoencoder latent vectors
  │
  └── Imagined speech EEG (1024 Hz, 6 ch)
        └── preprocessing (per-window z-score)
              ├── Handcrafted: DTWPT energy per EEG band (alpha/beta/theta)
              └── Learned: SNN-AE / ANN-AE latent vectors
                    │
                    ▼
            Paraconsistent evaluation (α/β → D_truth)
            Rank all (strategy × modality × scale) combinations
            Select best — no classifier trained yet
                    │
                    ▼
            Classifier (RNN or DSNN)
            Nested or flat 5-fold cross-validation
            Text-dependent + text-independent splits
                    │
                    ▼
            Results: accuracy, EER, D_truth per combination
```

### EER scoring strategies (`IEERScorer`)

EER computation is pluggable via `statistics::IEERScorer`:

```cpp
// Pass to run_classifier(); nullptr → GenuineImpostorEERScorer
statistics::GenuineImpostorEERScorer sota_scorer(/*n_enroll=*/1);
auto result = thesis::run_classifier(view, fvs, label, cfg, &sota_scorer);

// Back-compat alias — now delegates to the genuine/impostor method (audit m-4)
statistics::ClassificationEERScorer alias_scorer;
auto result2 = thesis::run_classifier(view, fvs, label, cfg, &alias_scorer);
```

| Scorer | Protocol | EER source | SOTA? |
|---|---|---|---|
| `GenuineImpostorEERScorer` | enroll N utterances → cosine similarity genuine/impostor trials → FAR/FRR sweep | Score distribution | ✓ |
| `ClassificationEERScorer` | upgraded (audit m-4): delegates to genuine/impostor with 1 enrollment utterance | Score distribution | ✓ |

`GenuineImpostorEERScorer` protocol:
1. Per speaker: first `n_enroll` utterances → L2-normalised mean embedding (template)
2. Remaining utterances → probes
3. Each probe scored vs. all templates: cosine similarity
4. Genuine trials (own speaker) vs. impostor trials (other speakers) → sort → threshold sweep → interpolated crossing = EER

See `include/statistics/eer_scorer.hpp` for the `ISplitPolicy`-style interface; add new strategies by implementing `IEERScorer::compute_eer(embeddings, labels, n_classes)`.

### Performance notes

**Parallel feature extraction** — `extract_features()` uses `#pragma omp parallel for schedule(dynamic, 4)` over samples. The DTWPT computation (`wavelets::malat`) has no global state, so parallelism is safe. Pre-sized `fs.vectors.resize(n_samples)` avoids `push_back` races; progress counter uses `#pragma omp atomic capture`.

**Pre-built dataset tensors** — `run_classifier()` builds `(N, D)` input and `(N, C)` one-hot target tensors once via `mutable_data_ptr()` before the outer fold loop. Each fold slices row views via `Tensor::row(idx)` — cheap view, no copy.

**True batch training** — `Trainer::fit_loop_supervised` stacks a `(B, D)` batch each iteration, doing one GPU kernel per layer instead of B tiny `(1, D)` kernels. Both `LinearImpl` and `CrossEntropyLoss` support arbitrary batch dimension. See [Training](../Core/Training.md#true-batch-supervised-training).

**Batch evaluate()** — test-set evaluation stacks all test samples into one `(N_test, D)` tensor, one `model.forward()` call, then inline argmax + one-vs-rest confusion matrix.

---

## Build and run

```bash
# Configure (once)
cmake --preset=max-performance

# Build experiment binary
cmake --build out/build/max-performance --target thesis -j$(nproc)

# Phase 00 — rank feature extractors per signal (stops after paraconsistent ranking)
./out/build/max-performance/src/experiments/thesis/thesis \
  --config src/experiments/thesis/profiles/phase00/p00_hc_daub4_lfcc_c1_voice.json

# Phase 01 — DSNN authentication with the chosen extractor
./out/build/max-performance/src/experiments/thesis/thesis \
  --config src/experiments/thesis/profiles/phase01/p01_dsnn_eeg_indep_nested_std.json

# Only repetition 1 of a profile: same seed and output names as in the full run. This is the
# unit the GridUnesp array runs. Any other argument is refused: the old parser ignored unknown
# ones, so "--repat 1" ran every repetition.
OMP_NUM_THREADS=1 ./out/build/max-performance/src/experiments/thesis/thesis \
  --config src/experiments/thesis/profiles/phase01/p01_dsnn_eeg_indep_nested_std.json --repeat 1

# Full article run (all real profiles: phase00 then phase01)
./scripts/testing/run_thesis_profiles.sh all
```

### Two-phase protocol

> **Running it:** step-by-step commands (build → phase00 → rank → apply → phase01, with the `run_thesis_profiles.sh` runner and live progress) are in [Running Experiment05 Profiles](../Guides/Running-Thesis-Profiles.md).

The experiment is split into two profile sets, gated by `classifier.enabled`:

- **Phase 00 — feature-vector construction** (`profiles/phase00/`, `classifier.enabled=false`, `paraconsistent.enabled=true`). For each signal (`voice`, `eegpron`, `eegimag`: the EEG under each condition, [§1](#1-the-eeg-was-the-previous-trials-eeg)), sweep the handcrafted extractor over **mother wavelet** (`handcrafted.wavelet`, 23 options with coefficient traits in `include/wavelet/Types.hpp`: `haar` + `daub4`…`daub46`) × **category** (`c1` energy / `c2` cepstral) × **scale** — `bark`/`mel`/`lfcc` for voice (138) but **`lfcc` only for EEG** (46; see the scale note below) = 184, plus the **compact AE sweep** — ANN-AE × `tiny`/`small`/`base` (6) and SNN-AE × `poisson`/`latency`/`direct` encoding × `tiny`/`small`/`base` (18) = 24 per signal, plus PCA k = 8/16/32: 153 voice + 61 per EEG condition = **275** rankings, and score every combination with the paraconsistent metric, on the whole set and on 30 speaker subsets ([§5](#5-how-stable-is-the-phase-00-winner)). The run stops after ranking — no classifier is trained, and `layer_spec` is not required. Output: paraconsistent CSV + summary JSON only. The lowest-`D_penalized` combination per signal wins; each family's best is kept too, for the comparison set.
- **Phase 01 — authentication** (`profiles/phase01/`, `classifier.enabled=true`, `paraconsistent.enabled=false`). The `feature_extraction` block is a placeholder until `02_thesis_apply_winner.py` writes the winners. Five sets (all generated): `p01_dsnn_*` the grid, 7 sources (voice, eegpron, eegimag, early/late fusion × condition) × text mode × CV scheme × raw/std = 56; `p01_ann_*` and `p01_cosine_*`, the controls, 7 × 2 text modes, flat, std = 28; `p01_dysph_*`, simulated dysphonia, the 5 voice-reading sources × scope × 4 levels = 40; `p01_cmp-<family>_*`, the best of each extractor family through the DSNN at the autoencoders' 100 epochs, 3 families × 3 signals × 2 text modes = 18. Total 142.

**Automating the Phase 00 → Phase 01 hand-off** — three scripts remove the manual seams, and
(since 2026-07-18) `run_thesis_profiles.sh phase00`/`all` chains all three itself once every
phase00 profile passes (fails closed on any failure, with no override; `THESIS_SKIP_POST`
disables it — see [Running Experiment05 Profiles](../Guides/Running-Thesis-Profiles.md)
for the full gating rules and the `scope=all` ordering caveat). Shown here for manual re-runs:

```bash
# Rank Phase 00 results, pick the min-D_penalized winner per signal (refuses a partial set):
python3 scripts/pipeline/thesis/01_thesis_phase00_rank.py \
  --profiles-dir src/experiments/thesis/profiles/phase00 \
  --results-dir  results/thesis/phase00 \
  --out          results/thesis/phase00/winners.json \
  --stability-out results/thesis/phase00/winner_stability.csv

# Inject the winners into the 142 Phase 01 profiles (--fused is required: no default):
python3 scripts/pipeline/thesis/02_thesis_apply_winner.py \
  --winners      results/thesis/phase00/winners.json \
  --profiles-dir src/experiments/thesis/profiles/phase01 \
  --fused        voice

# Regenerate the thesis Phase 00 tables from the ranked results:
python3 scripts/pipeline/thesis/thesis_build_phase00_paraconsistent_tables.py \
  --profiles-dir src/experiments/thesis/profiles/phase00 \
  --results-dir  results/thesis/phase00 \
  --tables-dir   ../../documentation/00-thesis/monography/tables
```

`01_thesis_phase00_rank.py` reads exactly the `e05_<run_tag>[_rep<k>]_paraconsistent.csv` files each profile declares (refusing a missing one, a CSV with more than one row, or one without `d_penalized`), selects on `d_penalized` (not raw `D_truth`, which a collapsed latent can game) per signal, reports exact ties explicitly, and writes `winners.json` (each winner carries its full `feature_extraction` block). `02_thesis_apply_winner.py` rewrites each Phase 01 profile's `feature_extraction` to its source's winner (`voice→voice`, `eeg` → the winner of its condition, `fused` → `--fused voice`, or with `--fused eeg` the winner of its condition; `p01_cmp-<family>_<signal>_*` → that family's winner); it refuses an autoencoder winner for a profile whose `training.epochs` is not the Phase 00 budget (100), before writing anything; it is idempotent and supports `--dry-run`. `thesis_build_phase00_paraconsistent_tables.py` regenerates the committed thesis tables from the profile set (every argument required; a missing summary or an empty `{ae, hc, pca} × {voice, eegpron, eegimag}` group is refused, and a result with no profile is never read). It ranks **ascending** by `d_penalized` (rank 1 = best) and marks each signal's overall winner (chosen across the handcrafted, autoencoder and PCA families) with a dagger — see pitfall 12 below for why this matters.

A fourth script, not yet chained into `run_thesis_profiles.sh`, ranks the Phase 01 results once that phase has run:

```bash
# Regenerate the thesis Phase 01 tables (grid, controls, dysphonia, extractors, energy):
python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py \
  --profiles-dir src/experiments/thesis/profiles/phase01 \
  --results-dir  results/thesis/phase01 \
  --tables-dir   ../../documentation/00-thesis/monography/tables
```

`thesis_build_phase01_auth_tables.py` averages `mean_eer`/`mean_auc` (and the networks' measured energy) across each configuration's 3 repeats and writes one CSV per set: `phase01_auth.csv` (the grid, ascending EER), `phase01_controls.csv`, `phase01_dysphonia.csv` (level 0 = the grid's clean cell), `phase01_extractors.csv` and `phase01_energy.csv` (with a `holds` column: whether the DSNN/ANN energy order survives every check). A profile whose name is in none of the five sets is refused, and each name is checked against its declared config. Every argument is required. The profile set defines the table: each profile's summaries must all exist and hold one result with a finite EER and AUC (else it refuses), the row label is built from the declared config (source, text mode, CV, standardization) and two profiles with one label are refused; with a single repetition the std columns print `—`, never 0.

### Running on GridUnesp (2026-11-12)

**Why.** Both phases run on the cluster, from the same binary and revision. Phase 01 carries
most of the work: 142 profiles × 3 repeats × 5 folds (cosine trains nothing), and since the
fold-intersection fix every fold trains on all of its training speakers, not about a third of
them.

**What runs where.** `scripts/pipeline/thesis/thesis_gridunesp.sbatch` runs one phase as a Slurm
array of **one-CPU** tasks. The work is cut into *units*: independent pieces that one process
runs from start to finish.

| Phase | Unit | Units | Array | Why this cut |
|---|---|---|---|---|
| 00 | one profile (all its repetitions) | 275 | `--array=0-63`, ~4–5 units per task | A handcrafted profile extracts its features once for all repetitions, because they do not depend on the seed. Split by repetition, it would extract them three times. The slowest profile took 230 s (array 5323006). |
| 01 | one (profile, repetition) | 142 × 3 = 426 | `--array=0-141`, 3 units per task | Repetitions are whole trainings with their own seeds. |

Task `t` of `N` runs units `t, t+N, t+2N, …` of the sorted list, one after another:

```
phase00, N = 64          sorted position:  0   1   2  ...  63   64   65  ...  213
                         task:             0   1   2  ...  63    0    1  ...   21
-> task 0 runs positions 0, 64, 128, 192 (4 profiles); the 24 autoencoder profiles
   (positions 0-23) land on 24 different tasks instead of all in one

phase01, N = 142         units: p01_ann_earlyimag_dep_flat_std.rep0, .rep1, .rep2, ...
                         task t runs units t, t+142, t+284:  thesis --config <profile> --repeat <k>
```

`thesis --config P --repeat k` runs repetition `k` exactly as the full run would: the seed, and
the `e05_<tag>_rep<k>_*` names. This was checked on 2026-11-12 with a 2-repetition, 2-epoch copy
of `p01_dsnn_voice_dep_flat_raw` at 1 thread. Repetition 1 run alone and repetition 1 of the
full run were compared, and every file was byte-identical except wall-clock fields (`train_ms`,
`infer_ms`, `epoch_ms`): EER/AUC, learning curves, spike counts and all 5 saved fold models.
Finer cuts were considered and not adopted:
- one process per outer fold would need a step that merges fold results into a summary, which
  the program does in memory today; not worth it for units that already take minutes;
- the inner-CV candidates of a nested fold still run one after another in a one-CPU process.

Why several units per task in Phase 00: every unit first loads the decoded dataset,
cache of its EEG condition (`~/database.sqlite.pronounced.e05dscache` or
`~/database.sqlite.imagined-2026.e05dscache`, 1.6 GB each), from `/home`. Units that run in a row on one node read
it once from `/home` and then from that node's page cache.

The local runner (`scripts/testing/run_thesis_profiles.sh`) follows the same rule. Each process
gets `THESIS_THREADS` threads (default 1; before 2026-11-12 every process took all the cores),
and as many processes run as memory allows.

**Why processes and not threads (measured 2026-11-12).** The program parallelises at three
levels. Two of them are coarse: whole inner-CV candidates, and the handcrafted extraction of
whole samples. The third is fine: xtensor's OpenMP and OpenBLAS split each multiply between
threads. The fine level dominates training, and on these small networks it loses. Job 5322998
ran one profile at each thread count on one node, at 1 and 3 epochs. The difference between
the two is the cost of 2 epochs, so loading and extraction are left out:

| Profile | 1 thread | 2 threads | 4 threads | per |
|---|---|---|---|---|
| `p01_dsnn_voice_dep_flat_raw` | 2 s | 2.5 s | 53 s | epoch (5 folds) |
| `p01_dsnn_eeg_dep_nested_raw` | 10.5 s | — | 14.5 s | epoch (5 outer folds, inner CV) |

node019 was also running four Phase 00 tasks of 4 CPUs each at the time. That is the situation an
array creates, and it is where the 4-thread penalty is largest: on that node, 4 CPUs are 2 cores ×
2 hyperthreads. On an idle 12-core desktop the same 3-epoch run took 30 s at 4 threads and 34 s at
1 thread, so the gain is small at best and is never worth a different result (below).

Phase 00 at 4 threads (array 5322982) showed the same effect from another angle. The 64-wide
`base` autoencoders needed minutes per repeat, while the 32-wide `small` ones finished all three
repeats in 20–38 s. A likely reading, not verified: xtensor parallelises only above a size
threshold, and only the 64-wide layers cross it.

**The thread count is part of the result.** That profile, after 3 epochs, gave EER 0.4176 /
0.4083 / 0.4201 at 1 / 2 / 4 threads. A parallel reduction adds in a different order, and
training amplifies the difference. The `.done` marker therefore records `threads=N` next to the
revision, and a phase directory holding another thread count is refused like one from another
revision.

The result is repeatable at a fixed thread count. On the desktop, two runs at 1 thread both gave
EER `0.42392311392251136`, and two at 4 threads both gave `0.4260507734969794`. The machine is
part of the result too: at 1 thread the same profile gave 0.4176 on node019 and 0.4239 on the
desktop. The two machines use a different CPU, and each was built with `-march=native` against
its own BLAS. So a phase runs entirely on the cluster, from one build, and its numbers are not
compared bit for bit with a local run.

This is the same lesson as meeting01 ([GridUnesp § Threads](../Guides/GridUnesp-Deployment.md#threads-why-20-cpus-made-the-run-about-1000-slower-2026-10-23)).
It also explains an earlier number: the "482 s for 1 epoch" measured locally ran at the 12
default threads.

**Commands.**

```bash
# local: sync the tree, build meeting01 + thesis on a compute node, send ~/database.sqlite
# (+ its two caches), and smoke thesis with p00_pca_k8_eegpron + _eegimag on a compute node
./scripts/pipeline/meeting01/gridunesp_deploy.sh            # GRIDUNESP_TARGETS="meeting01 thesis"
# cluster: one array per phase (# CONFIRMED -- the expensive-experiment guard)
THESIS_PHASE=phase00 sbatch --array=0-63 scripts/pipeline/thesis/thesis_gridunesp.sbatch
# local, when every task is done: bring the phase home, then the hand-off above
rsync -avz <user>@access.grid.unesp.br:software/nn/results/thesis/phase00/ results/thesis/phase00/
# after the hand-off is committed and deployed: Phase 01, one (profile, repetition) per task
THESIS_PHASE=phase01 sbatch --array=0-141 scripts/pipeline/thesis/thesis_gridunesp.sbatch
```

**What it refuses, and what it records.** Each line below is a loud stop. None of these cases
is skipped or worked around.

| Situation | Behaviour |
|---|---|
| A unit exits non-zero, or a summary / paraconsistent CSV that its repetition(s) must write is missing | That task stops at once with exit 1. `results/thesis/<phase>/.failed/<unit>.failed` holds the exit code, the missing files and the log path, and the task's later units do not run. |
| A unit finished and every expected file exists | `.done/<unit>.done` is written with the revision label and thread count. `<unit>` is `<profile>` in Phase 00 and `<profile>.rep<k>` in Phase 01. A resubmitted array skips that unit. |
| More tasks than units | The task is refused, because some tasks would have nothing to run. |
| A `.done` marker made by another revision anywhere in the phase directory | The task is refused. One phase directory must never hold two binaries' results, and nothing downstream could tell them apart. Move the directory aside. |
| The array is not `0..N-1` without gaps (`--array=3,7`) | The task is refused, because the profile-to-task mapping depends on `N`. Resubmit the whole array; finished tasks exit in seconds. |
| The binary was not built from the sources on disk (`SOURCE_REVISION.thesis.built` ≠ `SOURCE_REVISION`) | The task is refused. Rerun the deploy. |
| Time limit or `scancel --batch --signal=USR1` | The run is stopped and job-nanny copies the results back. The interrupted profile has no marker and reruns. |

The deploy stamps each binary separately and deletes every stamp before it builds, so a binary
it did not rebuild ends up unstamped and refuses to run. It also refuses to deploy while a
`thesis-run*` job is running: those jobs load this project's shared libraries from the build
tree (RPATH), and they read `~/database.sqlite`, which the deploy may replace.

All of the above was exercised locally on 2026-11-12, with stubs for `module`, `conda` and
job-nanny, using the real binary and 3 real profiles: a fresh task, a resubmission, a gap in
the array, a foreign revision, a failing profile, a call that is not an array, a stale stamp,
and a USR1 sent during an autoencoder profile.

The cache's temp file is now named `<cache>.tmp.<host>.<pid>`. The pid alone was unique on one
machine only, and array tasks on different nodes share `/home`.

**The first submission (array 5322956, 2026-11-12) failed on the database itself.** Six of 16
tasks stopped within 3 s with `Failed to prepare subject query`. `~/database.sqlite` is in WAL
mode, and WAL needs every reader on one host: its `-shm` index is shared memory. Every 10.1117
reader now opens the file immutable
([DataLoaders § Opening the database](../Core/DataLoaders.md#opening-the-database-immutable-because-of-wal-on-nfs-2026-11-12)).
This one failed loudly because the subject-list query already threw. The same failure one step
later, in a subject's row count, would have been read as zero rows until that morning's
no-fallback sweep. A subject would then have looked empty.

---

## Outputs

| File | Contents |
|---|---|
| `results/e05_*_metrics.csv` | Per-fold: accuracy, F1, precision, recall, EER, AUC, model_path |
| `results/e05_*_paraconsistent.csv` | α, β, G₁, G₂, D_truth per (strategy × modality × scale) |
| `results/e05_*_summary.json` | Config, seed, mean±std±ci95 for all metrics + per-fold model paths |
| `results/e05_*_comparison.dat` | pgfplots DAT: all aggregate metrics for thesis figures |
| `results/meeting01/models/<run_tag>/<feature_label>/fold_N.bin` | Trained model state dict per outer fold (binary, `nn::io` format) |

### Data fed to each profile family, and its metadata

Every profile pulls from the **same** paired audio+EEG trial set (`load_dataset` drops any trial missing either signal, so all modality runs are comparable — see [Dataset](#dataset)). What differs per profile family is *which signal(s)*, *what preprocessing*, and *what tensor shape* reaches the extractor:

| Profile family | Input signal(s) | Preprocessing | Shape fed to the extractor | Sample rate used |
|---|---|---|---|---|
| Handcrafted (`p00_hc_*`) | `sample.audio` or `sample.eeg` per `dataset.modality` | Voice: pre-emphasis (`α=0.97`). EEG: none. Zero-padded to next power of two for DTWPT. | Flat 1-D signal (voice: 176400 samples; EEG: 6 ch × 4096 = 24576, **channel-major flattened** — all of ch0, then ch1, …) | Voice 44100 Hz, EEG 1024 Hz (nominal, per Pressel Coretto et al. 2017) |
| ANN-AE (`p00_ae_ann_*`) | same as handcrafted, single signal | Log power spectrum, 256 bins (Welch, 512/256, Hann), **min-max normalized to `[0,1]`** (`spectral_input`, the same vector the SNN-AE and PCA read) | `(1, 256)` per sample, one forward pass | bins span 0 to Nyquist of the signal's own rate |
| SNN-AE (`p00_ae_snn_*`) | same as handcrafted, single signal | Log power spectrum, 256 bins, **min-max normalized to `[0,1]`** (`spectral_input`), then spike-encoded into `time_steps` frames (`poisson`/`latency`/`direct`) | `(1, 256)` **× T frames**, membrane state reset once per sample and integrated across all T frames | as ANN-AE |
| Fused-early (`*-fused-early`) | `sample.audio` ++ `sample.eeg` concatenated | Voice preprocessing applied to the whole concatenation (approximation — documented in code, not a physical resample) | Flat 1-D, voice length + EEG length | 44100 Hz applied to the whole signal |
| Fused-late (`*-fused-late`) | both signals, extracted independently | Each signal's own preprocessing above | Two independent extractions, vectors concatenated **after** | each signal's own rate |
| Phase 01 (`p01_dsnn_*`) | the Phase-00 winner's vectors (already extracted) | z-score standardization (`training.standardize_features`), fit on train folds only | `(N_samples, feature_dim)` | n/a — operates on feature vectors, not raw signal |

**EEG flattening order matters**: `ThesisSample::eeg` is `(N_channels=6, N_samples=4096)`; `tensor_to_vec` iterates rows-then-cols, so the flattened signal is channel-major (channel 0's full 4096-sample run, then channel 1's, …), **not** time-interleaved across channels. Anything reading a raw EEG feature vector must account for this layout.

**Per-sample dataset metadata** (`ThesisSample`, from `load_dataset`): `subject_id` (int, groups outer folds — GroupKFold, never split across train/test), `stimulus` (int, 1–10; mapped to `text_phrase` via `stimulus_to_phrase`: vowels `a/e/i/o/u` for 1–5, directional words `arriba/abajo/izquierda/derecha/adelante` for 6–10), `text_phrase` (string, drives `classifier.text_mode` dependent/independent splitting). These are **not** written per-row into the paraconsistent/metrics CSVs (which are aggregate-only); they are consumed internally by the fold-splitting and text-mode logic.

**Run-level metadata written to `summary.json`** (self-describing result files — added so a run's config doesn't have to be cross-referenced against its source profile):
- Always: `run_tag`, `seed`, `modality`, `strategy`, `classifier`, `text_mode`.
- `dataset.{n_subjects, n_stimuli, n_samples}` — the **actual** composition used (post audio+EEG pairing drop, can be smaller than the raw `.mat` trial count).
- `strategy=="handcrafted"` → `handcrafted.{wavelet, scale, cepstral, dtwpt_level, descriptors}`.
- `strategy=="autoencoder"` → `autoencoder.{model, encoder_layer_spec, decoder_layer_spec}`, plus for `model=="snn-ae"`: `encoding`, `time_steps` (forced to `1` when `encoding=="direct"`, regardless of the profile's configured value — the summary records what actually ran, not the raw field), `voltage_threshold`.

This closes the gap where all 18 SNN-AE profiles previously shared the identical FeatureSet label `"autoencoder-snn"` in the paraconsistent CSV — indistinguishable except by output *filename*. The label still doesn't carry encoding/size (labels are shared across the sweep by design, since paraconsistent ranking compares FeatureSets by label), but `summary.json` now does. Guarded by `ThesisOutput.*` gtests (handcrafted vs. autoencoder field presence, snn-ae `direct` forcing `time_steps=1`, ann-ae omitting snn-only fields, dataset composition roundtrip).

### Model checkpoints

After each outer fold, `run_classifier()` serializes the trained classifier state dict (DSNN or ANN; the cosine control has none) via `nn::io::save_state_dict(model.state_dict(), path)`. To reload an ANN checkpoint (`residual:2`, H = 128):

```cpp
#include "io/StateIO.hpp"
#include "ThesisAnnClassifier.hpp" // lib/src

thesis::ThesisAnnClassifier model(feat_dim, 128, n_speakers, /*depth=*/2, /*seed=*/42);
auto sd = nn::io::load_state_dict("results/meeting01/models/run/feat/fold_0.bin");
model.load_state_dict(sd);
```

Path pattern: `<results_dir>/models/<run_tag>/<feature_label>/fold_<N>.bin`

### Metric definitions

**Verification-only protocol (audit C-1).** Outer folds are speaker-disjoint
(GroupKFold by `subject_id`): test speakers are never seen in training. This is
an x-vector-style verification setup — train on background speakers, enrol the
unseen test speakers at evaluation time. Closed-set identification metrics
(accuracy / macro-F1 / precision / recall / specificity) are therefore **not
reported** (emitted as NaN); **EER and AUC are the primary metrics**.

| Metric | Formula | Notes |
|---|---|---|
| Accuracy / macro-F1 / P / R / specificity | — | **Not reported** (NaN) under verification-only: closed-set argmax over unseen speakers is invalid (audit C-1) |
| EER | FAR = FRR threshold crossing | **Primary.** `GenuineImpostorEERScorer`: cosine genuine/impostor trials |
| AUC | P(genuine_score > impostor_score) | **Primary.** Wilcoxon–Mann–Whitney; complement to EER |
| std_* | Sample std over folds | `sqrt(Σ(xi-μ)²/(n-1))` (audit M-3) |
| ci95_* | `t_{0.975,n-1} × std / √n` | Student-t CI for small fold counts (audit M-3) |

---

## Key differences from Experiment04

| Aspect | Experiment04 | Experiment05 |
|---|---|---|
| Purpose | Congress paper (SNN vs LSTM reconstruction) | Thesis primary experiment |
| Dataset | FSDD (spoken digits, 8 kHz, audio only) | 10.1117/12.2255697 (EEG 1024 Hz + voice 44100 Hz) |
| Task | Autoencoder reconstruction (MSE) | Speaker verification (EER, AUC; speaker-disjoint folds) |
| Signals | Audio only | Voice + EEG (bimodal) |
| Feature selection | Fixed architecture sweep | Paraconsistent α/β ranking before any classifier |
| Classifier | Autoencoder reconstruction loss | RNN / DSNN authentication |
| Text modes | N/A | Text-dependent + text-independent |
| Target population | General | Severely dysphonic speakers (DLS) |

---

## Common pitfalls

1. **Paired loading is always enforced.** `load_dataset()` always loads both audio and EEG for every trial, using the `eeg_index` column from the audio MAT to identify the correct EEG row. The `modality` field selects which signal is used during feature extraction — not which data is loaded. Subjects or trials missing either modality are silently dropped. This guarantees that voice-only, EEG-only, and fused runs operate on the **exact same set of subjects and trials**, making results directly comparable.

2. **Per-dimension [0,1] scaling before paraconsistent.** α and β are range-based and assume commensurable features in [0,1]. `score_feature_set()` min-max scales each dimension across all samples (audit M-1). Do **not** use per-sample sum-1 normalization — it lets large-magnitude descriptors (energy) dominate and pushes signed descriptors (Teager) outside [0,1], breaking the α domain.

3. **EEG and voice window sizes differ.** EEG at 1024 Hz needs different window parameters than voice at 44100 Hz. Do not reuse the same `window_size` across modalities.

4. **`jitter`/`shimmer` require voiced frames.** Unvoiced frames produce undefined period estimates. Filter by voicing flag before computing perturbation measures.

5. **Autoencoder path wires `snn-ae` and `ann-ae`.** `feature_extraction.autoencoder.model` is required and accepts `snn-ae`, `ann-ae`, or `lstm-ae`; the thesis Phase 00 profiles use `snn-ae`/`ann-ae` (`lstm-ae` is the legacy Meeting01 extractor, unused by any profile, and no longer the silent default it was until 2026-11-11).

6. **Text-independent split must not leak phrases.** Train and test splits must use disjoint phrase sets, not just disjoint utterances of the same phrase.

7. **Nested CV inner fold must not see test fold.** Hyperparameter selection must use inner-loop validation only.

8. **An unscorable fold throws; it used to be NaN.** `GenuineImpostorEERScorer` needs every speaker in the set to have a probe and at least two speakers. Until 2026-11-12 it returned NaN otherwise, the aggregate averaged over the folds that were not NaN, and an inner fold with one speaker scored −∞ and was never selected. Now the run stops and names the fold, its sample count and its speaker count (e.g. `outer fold 0, inner validation fold 2 (138 samples, 1 speakers)`).

9. **`max_samples` truncation round-robins across subjects.** Samples are stored subject-contiguous (~130 trials/speaker), so a first-N truncation would keep only 2–3 speakers and break speaker-disjoint (GroupKFold) folds — especially nested CV, whose inner fold would then have fewer groups than splits (`GroupKFoldPolicy: number of unique groups is less than n_splits`). `load_dataset()` therefore selects the capped subset round-robin across subjects so every speaker is represented. Found by the smoke suite — see [Ground-Truth and Smoke Testing](../Guides/Ground-Truth-and-Smoke-Testing.md).

12. **A ranking table's sort direction is not self-evident from the data — verify it against the actual winner.** `thesis_build_phase00_paraconsistent_tables.py` sorted `reverse=True` (descending by mean `d_truth`) for a long time before this was caught, on the reasoning that a ranked table "starts at the top". But `d_truth`/`d_penalized` are *distances* to the paraconsistent Truth vertex — lower is better — so descending order silently put each signal's actual winner **last**: the real EEG winner sat at rank 46 of 46. It also ranked on `d_truth`, not `d_penalized`, the metric `01_thesis_phase00_rank.py` actually selects on — the two orders genuinely diverge (see [Core/Paraconsistent.md](../Core/Paraconsistent.md#selection-metric-contradiction-penalized-truth-distance)). The fix: sort ascending by `d_penalized`, emit both `d_penalized` and `d_truth` columns, and mark the true winner (cross-checked against `winners.json`) with a dagger rather than trusting rank 1. When any script writes a "ranked" table, check its sort direction and key against an independent source of truth — do not assume rank 1 is the winner just because the column is populated.


13. **No fallbacks and no skips on the thesis path (2026-11-12).** Anything that cannot be computed stops the run with a message naming what and where; nothing is substituted or passed over. What used to be substituted or skipped, and now throws:

    | Was | Where |
    |---|---|
    | empty fold skipped (fewer folds than `k_folds`, run looked complete) | `require_fold_rows`, inner folds |
    | metric NaN in some folds averaged over the rest; 1 fold → SD 0 | `aggregate_metric` |
    | `t` quantile 1.96 outside the df 1–30 table | `t_crit_95` |
    | jitter/shimmer undefined → 0; ZCR/entropy/Teager on too-short or all-zero bands → 0 | `ThesisHandcraftedFeatures` |
    | `"energy"` declared with `cepstral = true` silently replaced | `ThesisConfig` (184 profiles migrated) |
    | unknown or repeated descriptor names ignored | `ThesisConfig` |
    | stimulus outside the table → phrase `"unknown"` | `stimulus_to_phrase` |
    | subject missing a modality, or with 0 rows, dropped | `decode_full_dataset`, `SubjectDiscovery` |
    | corrupt dataset cache decoded around; failed cache write ignored | `ThesisDataset` cache |
    | signal shorter than 256 stretched; constant signal → all zeros | `pool_signal` (replaced 2026-11-12 by `log_power_spectrum`, which refuses a signal under 512 samples and a zero-power bin), `normalize01` |
    | latency code with 1 step clamped; batch < 1 clamped | `spike_frame`, trainer config |
    | `layer_spec` items the code does not build ignored | `parse_layer_spec`, `AutoencoderBuilders` (pool1d / lstm / any SNN conv→residual) |
    | EER scorer: zero embedding → cosine 0; short speaker dropped; NaN | `eer_scorer.cpp` |

14. **`intersect_indices` silently lost training data (fixed 2026-11-12).** It was `std::set_intersection`, which requires both inputs sorted; a fold's train indices are other folds' buckets concatenated, not sorted. Flat-CV training sets lost 62 % of their samples on average (simulated on the dataset's real per-speaker sizes; 15–78 %), often keeping 3 of 12 training speakers; nested inner folds lost samples too. Every Phase 01 number before this date (EER **and** AUC) is from crippled training. Guarded by `ThesisIndexIntersection.*`.

15. **The EER sweep was inverted until 2026-11-12** — see [Core/Statistics.md](../Core/Statistics.md#sweep-direction-defect-fixed-2026-11-12--silent). The "standardization worsens EER while raising AUC" pattern of the 2026-07-20 Phase 01 was this defect: EER is threshold-free, so no calibration can make the two disagree.

16. **"base" and "small" autoencoders were the same network in the 2026-07-20 Phase 00.** The width helpers of that time took a width equal to their fallback (64 hidden, 16 latent) as "not found": base (64/32) and small (32/16) both ran as 32/32, identical to 8 decimals in all 6 pairs. Fixed 2026-07-23; guarded by `ThesisAeArchitecture.LatentWidthIsTheDeclaredLastEncoderWidth`.

17. **Nested CV now applies the text split to training.** It used to train on every non-test sample, including the phrases `text_mode = independent` holds out (pitfall 6), so its "independent" was not the flat path's. Outer train = (not outer test) ∩ text-train, as in flat CV.

18. **Stimulus names come from the database.** The `stimulus` table numbers 11 stimuli alphabetically (1 A, 2 Abajo, 3 Adelante, 4 Arriba, 5 Atras, 6 Derecha, 7 E, 8 I, 9 Izquierda, 10 O, 11 U). A hard-coded table assumed 10 (vowels 1–5, five words 6–10, no "atras") and named stimulus 11 `"unknown"`. The split only needs distinct groups, which it had; the names were wrong. Cache format bumped to version 2.

## Testing

Beyond the unit tests (`e05_*_gtest`), two extra layers guard this experiment:

- **Per-profile smoke runs** — `profiles/smoke/` mirrors all 241 profiles (208 phase00 + 32 phase01 + debug) with tiny run parameters; `scripts/testing/run_thesis_smoke.sh` runs each end-to-end to catch runtime errors compilation cannot. The mirror auto-regenerates via the CMake `thesis_smoke_profiles` target when any source profile changes.
- **scipy / PyTorch parity of the autoencoder and PCA input** — `thesis_spectrum_parity_gtest` (fixture from `scripts/testing/gen_thesis_spectrum_refs.py`): `spectral_input` against `scipy.signal.welch`, PCA against `torch.linalg.svd`.
- **PyTorch / snnTorch parity** — layer-level numerical ground truth (Linear, activations, MSE/CE losses, LSTM, LifBPTT, Conv1d/2d, MaxPool).

Both are documented in [Ground-Truth and Smoke Testing](../Guides/Ground-Truth-and-Smoke-Testing.md).

9. **SQLite float32 blobs.** The 10.1117 database stores audio/EEG blobs as `float32`. Loaders detect encoding by byte-size checks; mismatches throw explicit runtime errors.

10. **`nn::Tensor` default is a 0-dim scalar** (`size()=1, rows()=0, cols()=0`). Guard audio/EEG samples with `rows() > 0 && cols() > 0`, not `size() > 0`.

11. **`discoverSubjects` regex must have a capturing group.** Use `"^S(\\d+)$"` not `".*"`.

---

## See also

- [Paraconsistent Feature Engineering](../Core/Paraconsistent.md) — α/β/D_truth formulas and API
- [Paraconsistent — Plain](../Core/Plain/Paraconsistent.md) — accessible explanation
- [LFCC](../Concepts/LFCC.md) — frequency scale used in handcrafted features
- [Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) — neuroscience background
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md) — SNN-AE / DSNN theory
- [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md) — LSTM-AE theory
- [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) — nested CV
- [Data Loaders](../Core/DataLoaders.md) — 10.1117 loader API
- [Research Context](../Research-Context.md) — thesis goals and full pipeline
- [Experiment04](./Meeting01.md) — prior congress paper experiment
- [Re-run Runbook](../Guides/Re-run-Runbook.md) — commands to regenerate every result
- [Engineering Fixes Log](../Guides/Engineering-Fixes-Log.md) — the D1-D6 decision log behind the current `d_penalized` metric, 208-profile grid, and re-run
- [Experiment Microscope](../Guides/Experiment-Microscope.md) — inspect a Phase-00 run down to
  per-sample EEG channels and wavelet leaves; the GUI recomputes `load_dataset → extract_features
  → rank_feature_sets` via `nn_microscope.thesis`

---

## References

[A] R. C. Guido, "Paraconsistent feature engineering," *Knowledge-Based Systems*, 2018.

[B] S. Zhao et al., "EEG-based imagined speech recognition using deep learning," *IEEE Trans. Neural Syst. Rehabil. Eng.*, 2021.

[C] Dataset 10.1117/12.2255697: G. A. Pressel Coretto, I. E. Gareis, and H. L. Rufiner, "Open access database of EEG signals recorded during imagined speech," in *Proc. SPIE 10160, 12th Int. Symp. Medical Information Processing and Analysis*, 2017. [Online]. Available: https://doi.org/10.1117/12.2255697

[D] E. O. Neftci, H. Mostafa, and F. Zenke, "Surrogate gradient learning in spiking neural networks," *IEEE Signal Process. Mag.*, vol. 36, no. 6, pp. 51–63, 2019.
