# Proposal: dysphonia profiles beyond the whisper mix

**Status: PROPOSAL, nothing implemented.** Written 2026-11-15 after an exploration session
(literature, a WORLD prototype, four blind listening rounds). Current behaviour:
[Thesis § 4 Simulated dysphonia](./Thesis.md#4-simulated-dysphonia).

## 1. The problem

The thesis tests a system on dysphonic speakers, but no dysphonic voice exists in the data.
`VoiceDegradation.hpp` therefore *simulates* one: each 25 ms frame of a healthy voice is
re-excited with white noise through its own LPC envelope (a whisper), and the result is mixed
into the voice, `y = √(1−level)·x + √level·whisper`.

```
healthy voice ──► LPC envelope + energy ──► white noise through it ──► whisper w
y = sqrt(1-level) * x  +  sqrt(level) * w          level 0.25 | 0.5 | 0.75 | 1.0
```

This is the *only* dysphonia the code can produce, and the code has three weaknesses:

| Weakness | What it means | Loud or silent? |
|---|---|---|
| One kind of dysphonia | Breathy/aphonic only. Rough (irregular vibration), tremulous and broken voices cannot be made | silent: the title says "dysphonic", the experiment covers one type |
| `level` has no clinical unit | "level 0.5" is a power fraction; nobody can say which real voice it resembles | silent: every run produces an EER |
| Sounds artificial | One informal listener judged it "artificial" in 4 of 4 presentations (below) | silent: the classifier cannot hear it, the thesis text would not say it |

Today the whole experiment grid depends on it: **296 profiles declare `degradation`, 128 of
them enabled** (levels 0.25/0.5/0.75/1.0 × scopes `probes`/`test`).

## 2. What the exploration found

All listening below is **one listener (the author), informal, no repeated-measures design**.
Where the same sound was played twice, the naturalness answer agreed in 5 of 7 pairs, so a
single judgement carries roughly 30 % noise. The material was one 0.8 s run of the vowels
a-i-u-e-o (WORLD's own test file) and, for the main sessions, 2 s of a sustained pronounced
/a/ from Coretto (S10 male, 106 Hz; S03 female, 219 Hz).

| Question | Evidence | Verdict |
|---|---|---|
| Does a WORLD analysis→resynthesis with no change alter the voice? | natural + human in every presentation (short run, S10, S03) | transparent for this listener |
| Can WORLD produce tremor? | heard and human in 3 of 3 trials (4 % and 10 %; S10 ×2, S03). On the 0.8 s run it went unnoticed: the clip was too short | yes, on a sustained vowel |
| Pitch jumps / voice breaks? | heard, human (S10, S03); aphonic gaps heard but "human with defect" | yes; gaps need work |
| Jitter, shimmer, diplophonia, rough? | heard as "áspera" in 6 of 6 on S10 and 1 of 2 on S03 (S03 rough was heard as breathy); human in 11 of 12 S10 presentations, artificial in S03 jit10 | yes, on the male voice; one S03 miss |
| Breathy voice by WORLD (aperiodicity edit, pitch-synchronous noise, Klatt tilt + H1)? | stationary noise on S10: breathy and human; pitch-synchronous: human once, artificial once; S03: artificial both; tilt + H1: muffled and artificial | **not settled; no benefit from tilt, H1 or pitch-synchrony** |
| The current whisper mix? | artificial in 4 of 4 presentations | consistently judged artificial |

Acoustic checks (steady synthetic vowel, 140 Hz): jitter 1/3/6 % requested gave 1.14/2.63/5.46 %;
tremor 4/10 % gave 3.99/9.91 %; shimmer 1/3 dB gave 1.08/3.27 dB; the control moved jitter by
0.13 %. **On real vowels the realised jitter is lower than the request**: S10 1/3/6/10/15 %
gave 0.9/2.0/4.3/6.9/10.7 %, S03 lower still (the 1 ms frame limits per-cycle control at
high F0). Any number in the thesis must therefore be the *measured* one.

What the literature did and did not give (read in full: Morise 2016, Kreiman 2010, Klatt & Klatt
1990, Tayebi Arasteh 2023, Jiao 2018, Skowronski 2015, Lopes 2017 and, on 2026-11-15, Englert
2017, Kreiman & Gerratt 2005, Hillenbrand 1994 and 1996, Maryn 2010, Alku 1992, Fant 1985):
Kreiman's synthesizer is vowel-only and manual; Jiao needs
parallel patient data; Tayebi Arasteh's dysphonia result is confounded (microphone, age, sex)
and is not evidence either way; cepstral-peak values differ by about 15 dB between programs
(Skowronski 2015), so a severity scale needs one reference program.

## 3. The proposal

### 3.1 What stays untouched

`degrade_voice` and the numbers the whisper mix produces stay exactly as they are; the mix is
kept as the breathy/aphonic profile (`profile: whisper_mix`) and its limitation is documented,
not hidden. **Decision 1 (2026-11-15) changes one thing**: the new profiles are values of a new
`degradation.profile` key, so every profile that declares `degradation` must now declare
`profile: whisper_mix` (the no-defaults rule forbids leaving it out). `config_fingerprint`
hashes the declared keys (`thesis.cpp`), so the fingerprints of the 128 enabled profiles (64 in the manifest + their 64 `smoke/` mirrors) change
although their results do not. Consequences to handle when implementing: regenerate
`profiles/profiles.manifest` and re-run the profile audit; keep an explicit old-to-new
fingerprint table so results already on disk stay traceable; teach the parser's applicability
table (`for_each_field`) that `level` belongs to `whisper_mix` and the jitter/tremor keys to the
other profiles (a key the run does not read is refused).

### 3.2 New profiles (new values of `degradation.profile`)

Same logic as today (the TEST speakers' voices are replaced; training stays clean; `scope` =
`probes` | `test`; per-sample seed), a different generator:

| Profile | Parameters (measured units) | Status of evidence |
|---|---|---|
| `resynthesis_control` | none | transparent in every presentation |
| `rough` | `jitter_pct`, `shimmer_db` (local, as measured) | heard as rough, human |
| `tremor` | `f0_depth_pct`, `am_depth_pct`, `rate_hz` | heard, human |
| `voice_breaks` | `kind` = `pitch_jump` \| `aphonic`, `count`, `duration_ms` | pitch jump good; aphonic has a defect |
| `diplophonia` | `period_alt_pct`, `amp_alt_db` | heard as rough, human; jitter and shimmer are undefined for period-doubled signals (Kreiman and Gerratt 2005 excluded them for that reason), so report only the two alternation parameters |

Not proposed: a WORLD breathy profile (§ 6).

### 3.3 Shape of the change

```
ThesisConfig::Degradation  (+ new key `profile`)    "degradation": {enabled, profile, scope, seed, <keys of that profile>}
   profile = whisper_mix | resynthesis_control | rough | tremor | voice_breaks | diplophonia
validate: the keys declared must be exactly those the chosen profile reads; autoencoder refused, as today
VoiceDysphonia.hpp/.cpp  (new, beside VoiceDegradation)   degrade_voice_f0(x, native_rate, profile, seed)
cmake/VendorWorld.cmake  (new)  FetchContent pinned to a commit, like VendorCnpy.cmake
```

Decision 1 (2026-11-15): one block with a `profile` key, not a second block. The alternative
(a separate `dysphonia` block, mutually exclusive with `degradation`) would have left the 128
fingerprints untouched; it was declined in favour of a single place for every degradation. The
cost is the fingerprint change described in § 3.1. The project rule (no defaults, no fallbacks)
forbids treating a missing `profile` as `whisper_mix`.

WORLD (`mmorise/World`): modified-BSD licence (three clauses: keep the notice, no endorsement),
README states "no patent in all algorithms". Its `Synthesis()` takes an editable D4C
aperiodicity matrix (the 2016 paper's "cannot manipulate aperiodicity" describes the old
PLATINUM version). Citation requested by the README: "WORLD [1] (D4C edition [2])".

### 3.4 Severity in measured units

Request a target, run, measure, correct, as a closed loop in the calibration test (not at run
time):

```
requested jitter 3 %  ──► synthesis ──► measured 2.0 % (S10)   ──► scale factor ≈ 0.67
```

Each profile states the *measured* jitter/shimmer/depth it produces on the real vowels of each
base. The estimator (zero crossings of the low-passed signal, cross-checked by pulse-peak
instants) is part of the test code. Cepstral peak prominence (CPPS) is **not** proposed as
the severity axis of these profiles: Hillenbrand and Houde (1996) validated it for voices that
depart from normal mainly towards *breathiness* (92 % of the variance for sustained vowels) and
say it is not known whether it predicts roughness; Maryn et al. (2010) likewise describe
cepstral measures as mainly associated with breathiness and less with roughness. Two further
points from Kreiman and Gerratt (2005): listeners agreed on the overall noise-to-signal ratio
(about 10.7 dB difference needed to be heard 75 % of the time) but not on jitter or shimmer
(about 2.3 % jitter and 1.4 dB shimmer without noise; with noise, never reliably heard up to 3 %
jitter), so every profile's output should also report its harmonics-to-noise ratio next to the
jitter, shimmer or depth; and the same paper reports, second hand from Hillenbrand (1988) (not
read), that perceived roughness stopped rising beyond about 2 % jitter and 2 dB shimmer, so
jitter levels of 4–7 % test the verification system, not a perceived roughness (the thesis
makes no perceptual claim, decision 3). If CPPS is ever used for a breathy profile, one program
must measure real and simulated voices alike (Skowronski 2015: about 15 dB between programs;
Maryn 2010 itself took CPPS from SpeechTool and the other five AVQI terms from Praat).

#### 3.4.1 Proposed levels (provisional: each is a *measured* target, reached by the closed loop above)

Levels are chosen from three constraints: the measurable range (S10 realised jitter 0.9/2.0/4.3/6.9/10.7 % for 1/3/6/10/15 % requested; S03 lower), the perceptual range of Kreiman and Gerratt (2005: scales 0–3 % jitter and 0–2 dB shimmer; limens about 2.3 % and 1.4 dB without noise), and where our own listening heard the effect. Nothing here claims a perceived severity (decision 3): the low levels are a dose-response start near the control, not "mild".

| Profile | Levels (measured) | Basis | Open |
|---|---|---|---|
| `resynthesis_control` | 1 | no parameter | identity pilot (§ 4.3) |
| `rough` | jitter 0.5 / 1 / 2 % with shimmer 0.5 / 1 / **1.5** dB, plus robustness 4 % and 7 % (shimmer **1.5** dB; was 2 dB, unreachable as measured, decision 7) | 0–2 % is the range Kreiman and Gerratt scale and where roughness still rose (Hillenbrand 1988, second hand); 4 % and 7 % were realised on S10 (nominal 6 and 10) | audibility below about 2 % was never isolated in our listening; shimmer realised on real vowels was not measured; 7 % may not be reachable on S03 |
| `tremor` | F0 depth 2 / 4 / 10 %, rate 5 Hz | 4 % and 10 % heard and human in 3 of 3 trials; Kreiman and Gerratt's tremor rates averaged 6.45 Hz (median 6, range 2–10) | 2 % untested; the amplitude depth should follow from the F0 depth, not be a free knob: the same paper notes many "amplitude tremors" are artefacts of frequency modulation (Sundberg 1995; Kreiman 2003), so `am_depth_pct` becomes a fixed ratio, not a reported unit |
| `voice_breaks`, `pitch_jump` | 0.5 / 1 / 2 jumps per second of voiced speech, factors ×1.5 and ×0.65, 80 ms | heard and human on S10 and S03 | utterance length differs by base, hence a rate, not a count per utterance |
| `voice_breaks`, `aphonic` | same three rates, 80 ms gaps | heard, "human with defect" | the defect is unresolved; weakest evidence of the five |
| `diplophonia` | period alternation 6 / 12 % with amplitude alternation 3 / 6 dB | heard as rough, not told apart from jitter | jitter and shimmer are undefined here; report only the two alternation values |

Counts: 1 + 5 + 3 + 3 + 3 + 2 = 17 levels, against 4 for the whisper mix; with the two scopes that is 34 combinations per base against 8 today. The re-run cost is not estimated, and the identity pilot (§ 4.3) should come first. Every level also reports the harmonics-to-noise ratio of its output (§ 3.4), and the calibration test records which levels are not reachable on which voice instead of clamping them.

#### 3.4.2 Calibration result (measured 2026-11-15)

**How.** `scripts/pipeline/thesis/05_thesis_dysphonia_calibration.py` compiles the thesis's own
`VoiceDysphonia.cpp`, degrades every pronounced vowel trial of S10 (69 trials, mean F0 100 Hz)
and S03 (65, 205 Hz), and measures raw and degraded versions with **Praat 6.1.38 through
Parselmouth 0.4.7** (Boersma and Weenink 2021; Jadoul et al. 2018) on the longest voiced run
of the raw trial. Sweep each axis, read the request whose *mean* measured value is the target,
re-run at that request and count the voices within ±15 %. 134 trials, none excluded, 47 min
on 10 processes. Output: `results/thesis/dysphonia_calibration/` (`calibration.json`,
`measurements.csv`, one row per voice and request).

| What | Read back by Praat | Verdict |
|---|---|---|
| `resynthesis_control` | RMS within 1 dB in 134/134; HNR **+2.5 dB (S10), +3.8 dB (S03)**, within ±1.5 dB in 4/69 and 0/65; jitter 0.39→0.29 % and 0.45→0.26 %, shimmer 0.23→0.17 and 0.24→0.13 dB | **fails** the acoustic criterion of § 4 item 1: resynthesis makes the voice *more periodic* than the recording |
| `rough` jitter | ≈0.65 of the request (6 % → 3.9 %), monotone to 8.6 % at 15 %, both speakers alike | calibrated: 0.5 % ← request 0.62 (102/134 within ±15 %); 1 % ← 1.47 (101/134) |
| `rough` shimmer | ≈1.0 of the request up to 1 dB, then **saturates at 1.95 dB** | 0.5 dB ← 0.47 (129/134); 1 dB ← 0.99 (132/134); **2 dB unreachable as measured**: Praat's standard maximum amplitude factor 1.6 (4.1 dB) discards the pairs that differ more, so the mean of the kept pairs cannot pass ~2 dB. Not a generator limit: with the factor at 3.0, S10's amplitude alternation reads exactly up to 8 dB |
| `tremor` | 0.98 of the request | 2 % ← 2.04 (124/134); 4 % ← 4.08 (134/134); 10 % ← 10.21 (134/134) |
| `voice_breaks` | 1.08–1.22 × the requested rate (density over the longest voiced run, which excludes the edges) | rates realised; no calibration needed |
| `diplophonia` | S03: Praat locks on the subharmonic (~102 Hz) and reads ~0 alternation (reproduced on a synthetic 205 Hz vowel); S10: 0.46 of the request (a synthetic 100 Hz vowel: 0.85–0.97) | **not validated.** On high voices the reference measurer cannot read it (the literature's "jitter is undefined for diplophonia"); on S10 the cause of the loss is not identified (ruled out on synthetic vowels: level, F0 drift, aspiration noise, high-pass, source tilt). Kept out of any grid until resolved |

Consequences for § 3.4.1: the three `rough` levels with 2 dB shimmer are unreachable as
specified (their jitter targets are reachable: 7 % ← request ≈11.2); both `diplophonia` levels
are out. Options for the user: shimmer 1.5 dB for those levels (reached at request ≈1.7), or
report the 2 dB by the requested value with the ceiling stated. The failed acoustic control
does not by itself block the grid: every profile is compared with the control, not the raw
voice (§ 3.5); the identity pilot (§ 4.3) decides whether the control keeps the speaker.

**Re-run 2026-11-16 with the decided levels (decision 7: shimmer 1.5 dB; decision 8: no
diplophonia), 48 min.** Every level reachable on the mean (`unreachable_levels` empty). The
three strong `rough` levels: shimmer 1.5 dB ← request 1.71, read back 1.49–1.57 dB, 130/134
trials within ±15 %; jitter 2 % ← 3.05 (97/134), 4 % ← 6.17 (92/134), 7 % ← 11.23 (103/134).
The other levels and the control repeat the first run (same seeds, same voices). The WORLD grid
(§ 4.5) reads its requests from this `calibration.json`.

### 3.5 The control arm

A WORLD profile changes the signal even with no dysphonia (resynthesis). Every grid that uses a
WORLD profile therefore also runs `resynthesis_control` over the same utterances; the dysphonia
effect is the difference between the profile and the control, never between the profile and the
raw voice. The whisper mix has no such issue (it keeps the original waveform).

### 3.6 Errors, not fallbacks

| Situation | Behaviour |
|---|---|
| analysis rate ≤ 8 kHz | `std::invalid_argument`: D4C has 0 bands there and returns aperiodicity 1 everywhere (a whisper). Observed: `fs=8000`, aperiodicity 1.000 |
| utterance with no voiced frame | error naming the sample (no silent pass-through) |
| output length ≠ input length | resampled to the same length or error, never truncated silently (`degrade_voices` copies `y` into the tensor) |
| BIOMEX voice | analysed at its native 16 kHz (`recording.voice_native_rate`), not the 44.1 kHz it is stored at: at 44.1 kHz D4C would estimate bands 6–15 kHz from an empty spectrum (read from the code, not run) |

## 4. Validation plan

1. **Acoustic (unit tests, no thesis binary):** control within 1 dB RMS and 1.5 dB HNR of the
   input; measured jitter/shimmer/tremor within ±15 % of the calibrated target on a steady
   synthetic vowel and on the Coretto vowels; bit-identical output for the same seed;
   thread-independence under OpenMP (WORLD's internal random state looks local, **unverified**).
2. **Perceptual: none in the thesis (decision 3, 2026-11-15).** No more listeners are available,
   and the listening so far is one informal listener (the author), so the thesis makes no
   perceptual claim: it describes each profile by the measured jitter, shimmer or depth it
   produces. Literature checked for this decision: Lopes et al. 2017 (871 VoiceSim vowels, one
   expert rater, intra-rater kappa 0.88, no external validation); Englert et al. 2017 (J Voice; 269
   listeners invited, 99 dropped for answering the same way on fewer than 72.2 % of repeated
   items, 170 analysed: 58 voice-specialist SLPs, 51 general SLPs, 61 naive; 18 human against 18
   VoiceSim voices; 39.3 % misclassified overall, 42.3 % of the synthesized and 36.4 % of the human
   voices, while the paper's Table 2 total reads 39.8 %); Kreiman and Gerratt 2005 (jitter and
   shimmer are not useful as independent indices of perceived quality; 70 naive listeners in
   experiment 1, 18 in experiment 2); no study found that joins simulated dysphonia, listening
   tests and speaker verification. Two facts from Englert matter for the author's own listening:
   real dysphonic voices were taken for synthetic 36.4 % of the time (52.1 % for the severe ones),
   and the retained listeners agreed with themselves on 81.6 % of repeated items (kappa 0.58),
   so one listener calling a voice "artificial" is weak evidence (the author's own repeat
   agreement was 5 of 7). A jitter value is not a perceived severity, so the text must not use
   "mild/moderate/severe" for it.
3. **Identity (decisive; decision 4, 2026-11-15: pilot first):** does `resynthesis_control`
   change the speaker-verification EER on a base? After implementation: time one debug profile,
   report the cost, and run only with the user's explicit confirmation through the
   `expensive-experiment-guard` hook (Coretto only, raw against control, smallest configuration).
   The full grid starts only if the control is transparent.

   **Pilot design, fixed before any pilot result (2026-11-15; user confirmation the same day).**
   Profiles `pilot/pil_<kind>_voice_{raw,control_probes,control_test}` (generated; results in
   `results/thesis/pilot/`): Coretto voice, dependent text, flat 5-fold CV, standardized, 3
   repeats (seeds 42-44), the Phase 00 voice winner. `kind` = `cosine` (no training: the
   only thing that differs between arms is the test features, so it isolates what resynthesis
   does to the speaker) and `dsnn` (the cell of `p01_dysph_*`, trained on clean voices in every
   arm). Unit: the EER of one (repeat, fold); ΔEER = control − raw, paired on the same split.
   Margin: **±0.01 EER**, half the smallest whisper-mix effect worth reporting (committed
   Phase 01 DSNN results: mean ΔEER +0.007 at `probes` l25, +0.013 to +0.041 elsewhere; per-pair
   sd 0.02-0.06), because a margin as wide as the effect would let a control that moves the EER
   as much as the mildest dysphonia pass as neutral.
   - *Transparent:* the 95 % interval of the mean ΔEER over the 15 pairs lies inside ±0.01
     **and** each repeat's mean ΔEER does too, for both scopes, on `cosine`. The interval is the
     corrected resampled t of Nadeau and Bengio (2003), variance × (1/15 + 1/4), df 14: the
     test every other comparison of the thesis uses (ch. 8, "Protocolo de treino"). *Changed the
     same day, before any pilot result:* the first version of this rule used the plain paired t,
     about 2.2 times narrower, which the thesis does not use for 15 dependent pairs.
   - *Not transparent:* the interval lies entirely outside ±0.01 for a scope.
   - *Inconclusive:* anything else.
   The 15 pairs share speakers and training data and are not independent; the correction widens
   the interval for that, and each repeat must also pass. `dsnn` is reported with the same statistics but decides
   nothing on its own: with per-pair sd ~0.04 its corrected interval is about ±0.05 wide
   before any effect. The voice winner reads jitter and shimmer descriptors, and resynthesis lowers the
   measured jitter (calibration: S10 0.42 → 0.25 % on one vowel), so a non-transparent
   result is plausible and is reported as such, not tuned away.

   **Pilot result (run 2026-11-15, 42 min locally; `results/thesis/pilot/pilot_verdict.json`,
   `06_thesis_identity_pilot.py`): INCONCLUSIVE.**

   | arm | raw EER | control EER | ΔEER | corrected 95 % interval | repeats > 0 | verdict |
   |---|---|---|---|---|---|---|
   | cosine, `probes` (decides) | 0.3821 | 0.4003 | +0.0182 | [−0.0227, +0.0591] | 3/3 | inconclusive |
   | cosine, `test` (decides) | 0.3821 | 0.3894 | +0.0073 | [−0.0327, +0.0472] | 2/3 | inconclusive |
   | DSNN, `probes` | 0.3199 | 0.3669 | +0.0470 | [−0.0194, +0.1133] | 3/3 | inconclusive |
   | DSNN, `test` | 0.3199 | 0.3668 | +0.0469 | [−0.0403, +0.1341] | 3/3 | inconclusive |

   What it means. (1) **The pilot could not have passed**: the corrected interval's half-width
   (~0.04 on cosine) is four times the margin, so even a null effect reads inconclusive with
   15 dependent pairs. That is a design flaw of the pilot, found by running it, not a property
   of the control. (2) Every estimate points the same way: resynthesis raised the EER in 11 of 12
   repeat means, and with the DSNN by +0.047, more than the full whisper in the same cell
   (+0.041, Phase 01 `probes` l100). Suggestive, not significant. (3) Under the rule, the grid
   does **not** start. The raw DSNN arm reproduces the committed Phase 01 cell (0.3199 vs 0.3180).

   Ways forward (the user's decision): a test with power for a ±0.01 margin (more repeats, or a
   paired per-utterance measure: the cosine similarity of each utterance's raw and
   resynthesized feature vectors against the same-speaker spread, n = 1974 instead of 15); or
   accept a non-neutral control, keep "profile minus control" (§ 3.5) and state the control's
   own shift; or drop the WORLD profiles from the thesis.
4. **Identity, powered (decision 6, 2026-11-15).** *Rule fixed and committed before any result
   of this test.*

   *Why the pilot could not decide.* An EER pools every utterance of a fold into one number,
   so the pilot had 15 numbers per arm, all sharing speakers. This test keeps the utterance.

   *The measure.* Features of the cosine pilot (the Phase 00 voice winner: DTWPT-LFCC, haar,
   level 4, with energy, ZCR, entropy, Teager, jitter and shimmer descriptors), standardized
   with the mean and sd of the clean features. For utterance u of speaker s saying word w:

   ```
   own(u)     centroid of s's OTHER clean utterances of w        (u left out)
   other(s')  centroid of s''s clean utterances of w,  s' != s   (14 rivals)
   margin(f)  cos(f, own(u)) - max over s' of cos(f, other(s'))
              > 0: f lies nearer its own speaker than the closest impostor

   d_control(u) = margin(control features of u)     - margin(clean features of u)
   d_whisper(u) = margin(whisper l25 features of u) - margin(clean features of u)
   ```

   *The unit.* Each of the 15 speakers gives ~130 dependent utterances, so the inference is on
   the 15 speaker means of d: Student t, df 14, 95 % interval. 1974 utterances as if independent
   would be pseudoreplication.

   *The yardstick and the margin.* Whisper l25 (`p01_dysph_voice_probes_l25`, the mildest
   degradation of the grid). Its interval of mean d_whisper must lie below 0; if it does not,
   the yardstick is void and there is **no verdict** (the script stops with an error, nothing is
   tuned). Margin δ = |mean d_whisper| / 2, half the mildest whisper effect, the same reasoning
   as the pilot's ±0.01.
   - *Transparent:* the 95 % interval of mean d_control lies inside (−δ, +δ).
   - *Not transparent:* the interval lies entirely outside (−δ, +δ).
   - *Inconclusive:* anything else.

   Features: `thesis_features --config <profile>` (`tools/thesis_features.cpp`, the same
   `load_dataset` / `extract_features` / `degrade_voices` the thesis binary runs), on
   `pilot/pil_cosine_voice_control_probes.json` and `phase01/p01_dysph_voice_probes_l25.json`;
   their clean features must be identical (checked). Analysis:
   `scripts/pipeline/thesis/07_thesis_identity_margin.py` →
   `results/thesis/identity_margin/identity_margin_verdict.json`. What the verdict permits:
   transparent → the WORLD profiles (without diplophonia) may enter a grid, with the user's
   confirmation; not transparent → the control's own shift is reported and every WORLD profile is
   read only as "profile minus control" (§ 3.5); inconclusive → no grid, the thesis says so.

   **Result (run 2026-11-15 after the rule's commit 8ffdf9b93; features 490 s + 551 s locally;
   `results/thesis/identity_margin/`): NOT TRANSPARENT.**

   | | mean Δmargin (15 speaker means) | 95 % interval | speakers < 0 |
   |---|---|---|---|
   | whisper l25 (yardstick) | −0.0273 | [−0.0414, −0.0131] | — |
   | `resynthesis_control` | **−0.0976** | [−0.1523, −0.0429] | 14 / 15 |

   δ = 0.0136. The clean margin averages 0.127 (positive for 83.6 % of utterances). The control
   lowers it **3.6 times as much as the mildest whisper**, and its whole interval lies below −δ.

   *Exploratory, after the verdict (changes nothing in it; the `exploratory` block of the JSON):*
   per descriptor, the mean signed shift in clean-sd units, control / whisper l25: shimmer
   +0.92 / −0.32, entropy −0.28 / +0.06, jitter −0.21 / −0.05, ZCR +0.10 / +0.05, energy and
   Teager ≈ ±0.02. Without the 32 jitter and shimmer columns, neither shift is distinguishable
   from 0 (control −0.0088 [−0.0266, +0.0090]; whisper −0.0015 [−0.0073, +0.0044]). So the
   speaker information resynthesis disturbs sits in the same per-band peak-irregularity
   descriptors through which the whisper acts, and which the dysphonia profiles would edit (these
   are the extractor's band descriptors, not Praat's jitter/shimmer).

   Consequence (the rule): no WORLD profile enters a grid unless read as "profile minus
   control"; the thesis (ch. 8) reports the control's own shift and keeps the dysphonia
   conclusions of ch. 9 to the whisper model. A grid of "profile minus control" cells would be
   a new, expensive run: the user's decision.
5. **The WORLD grid (decided 2026-11-16, one question at a time): run it, mirror `p01_dysph_*`
   on both bases, confirmatory reading on the per-utterance margin, on GridUnesp.** *Rule fixed
   and committed before any result of the grid* (`scripts/pipeline/thesis/08_thesis_world_analysis.py`
   holds it in code).

   *Why "profile minus control".* § 4.4 measured the control's own shift. A WORLD level compared
   with the original voice would add that shift to the dysphonia's, so every comparison is
   level − control, both through the same WORLD path:

   ```
   original --WORLD-------------------------> control   (resynthesis: measured in § 4.4)
   original --WORLD + edited F0/amplitude---> level L   (resynthesis + dysphonia L)
   L − control                                           (dysphonia L alone)
   ```

   *The grid.* The cell of `p01_dysph_*` (DSNN, dependent text, flat 5-fold CV, standardized,
   3 repeats, Phase 00 winners): every source that reads the voice × scope (probes, test) ×
   (control + 14 levels). Coretto: voice, early/late × pron/imag = 5 sources, 150 profiles
   (`profiles/world/p01w_*`); BIOMEX-DB: voice, earlypron, latepron, 90 profiles
   (`profiles/bxworld/bx01w_*`). Levels: rough jitter/shimmer 0.5/0.5, 1/1, 2/1.5, 4/1.5,
   7/1.5 (% / dB, measured targets); tremor 2, 4, 10 % at 5 Hz; pitch jumps and aphonic gaps of
   80 ms at 0.5, 1, 2 per second. The rough and tremor requests are read from the re-run
   `calibration.json` by the generator (`--calibration`), never typed. No diplophonia (decision 8).

   *Part A, confirmatory (the margin of § 4.4, unchanged definition).* For each level L, on
   source voice: d_L(u) = margin(L features of u) − margin(control features of u); the speaker
   means of d_L (Coretto 15, BIOMEX-DB 45); two-sided one-sample t against 0, df = speakers − 1;
   Holm over the 14 levels of a base. L **changes identity beyond resynthesis** on a base when
   its Holm p < 0.05 (the sign says which way). BIOMEX-DB is confirmatory: L **replicates** when
   it is significant on both bases with the same sign. Features: `thesis_features` on
   `world/p01w_voice_probes_<level>` and `bxworld/bx01w_voice_probes_<level>` (the tool degrades
   every voice, so the scope does not matter), into `results/thesis/world_margin/`; the clean
   features of the 15 runs of a base must be identical (checked).

   *Part B, descriptive.* The EER of all 240 profiles, each level against the control of its own
   (source, scope), corrected t (Nadeau–Bengio) on the 15 (repeat, fold) pairs, Holm over the 28
   comparisons of one (base, source). Reported, never used for "changes identity": the pilot
   showed this interval to be ~±0.07 EER.

   *What the result permits.* Only statements of the form "level L moves the voice away from /
   towards its speaker beyond what WORLD resynthesis does, by this much", per base, with the
   measured acoustic value of L (calibration) beside it, never a perceived severity (decision 3).

   **Result (2026-11-17; arrays 5328285 world + 5328575 bxworld, 150 + 90 profiles, no failure;
   `results/thesis/world_analysis/`, thesis ch. 9 `sec:resultadosWorld`, table `tab:worldMargin`).**
   Level − control on the per-utterance margin, speaker means, Holm over 14 per base:

   | Level | Coretto Δ (15 speakers) | Holm p | BIOMEX-DB Δ (45 speakers) | Holm p |
   |---|---|---|---|---|
   | rough 0.5 % / 0.5 dB | −0.0020 | 0.030 | +0.0001 | 1 |
   | rough 1 % / 1 dB | −0.0068 | 0.003 | +0.0006 | 1 |
   | rough 2 % / 1.5 dB | −0.0182 | <0.001 | −0.0012 | 1 |
   | rough 4 % / 1.5 dB | −0.0184 | <0.001 | −0.0001 | 1 |
   | rough 7 % / 1.5 dB | −0.0173 | <0.001 | +0.0001 | 1 |
   | tremor 2 / 4 / 10 % | 0.0000 / −0.0014 / **−0.0143** | 1 / 0.599 / 0.021 | −0.0013 / −0.0010 / −0.0035 | 1 / 1 / 0.186 |
   | pitch jumps 0.5 / 1 / 2 per s | +0.0001 / +0.0007 / +0.0027 | 1 / 1 / 0.082 | +0.0018 / +0.0018 / +0.0028 | 1 / 1 / 0.746 |
   | aphonic gaps 0.5 / 1 / 2 per s | **+0.0087 / +0.0142 / +0.0275** | all <0.001 | −0.0056 / −0.0056 / −0.0095 | 1 / 1 / 0.645 |

   * Coretto: 9 of 14 levels change identity beyond resynthesis. Roughness lowers the margin,
     growing from 0.5 to 2 % jitter, then flat (~−0.018: the same saturation near 2 % that the
     listening literature reports, see § 2). Only the strongest tremor matters. Aphonic gaps
     *raise* the margin relative to the control (a relative sign: the control itself lowered the
     margin by −0.098; untested hypothesis: each gap replaces cycles resynthesis would distort).
     The strongest roughness effect is smaller than the mildest whisper on the clean voice (−0.027).
   * BIOMEX-DB: no level significant, so **no level replicates**.
   * Part B (EER): 0 of 224 comparisons significant (min Holm p 0.605), max |Δ EER| 0.016.
   * *Exploratory, after the result* (the `exploratory` block of `world_margin.json`; changes
     nothing in the verdict): BIOMEX-DB's control moves the voice away from its speaker too
     (control − clean −0.0648 [−0.0762, −0.0534], 43 / 45 speakers below 0) and leaves 13 % of
     the clean margin (0.0745 → 0.0097), against 25 % on Coretto (0.127 → 0.032). A floor effect
     on BIOMEX-DB is plausible, not tested.

6. **The floor test (decided 2026-11-17, after the grid result: "pré-registrar e testar").**
   *Rule committed before the run* (`scripts/pipeline/thesis/09_thesis_world_floor.py`, its
   docstring). Not a blind pre-registration: the grid result and the exploratory 13 % / 25 % were
   already seen; the per-speaker relation below was not.

   *Prediction.* If BIOMEX-DB's null is a floor, the speakers who keep more margin after
   resynthesis ("room") show a stronger (more negative) roughness effect.

   *Why split.* Regressing d = margin(L) − margin(control) on margin(control) is biased towards
   the predicted negative slope: the same noisy margin(control) sits in both with opposite signs
   (regression to the mean). So room and effect come from disjoint utterances, centroids
   included:

   ```
   each (speaker, word), in sample order:  u1 u2 u3 u4 u5 u6 ...
                                            A  B  A  B  A  B
   r = ( corr(room_A, effect_B) + corr(room_B, effect_A) ) / 2      over the 45 speakers
   ```

   effect = mean over the five roughness levels of margin(L) − margin(control). Two-sided
   permutation test of the speakers' rooms (100000, seed 20261117). Floor **supported** when
   p < 0.05 and r < 0; **contradicted** when p < 0.05 and r > 0; otherwise not supported. Only
   roughness (the family with a clear Coretto effect; a floor limits decreases). Power: 80 % only
   for |r| ≥ ~0.41 at 45 speakers, so a null does not exclude a weaker floor. Coretto: the same
   numbers, descriptive.

## 5. Cost and risk (not measured)

- Runtime of WORLD analysis per utterance, and the memory of the 1 ms frame matrices on
  4-second Coretto trials: **unmeasured**; the prototype ran 0.8–2 s clips only.
- New grid cells (profile × level × scope × bases) need re-running; I did not estimate it.
- The 1 ms resampling limits per-cycle jitter at high F0 (S03 realised the least); a finer
  frame period raises memory.
- Perceptual evidence is thin (one listener). Seven papers that were missing (Englert 2017,
  Kreiman and Gerratt 2005, Hillenbrand 1994 and 1996, Maryn 2010, Alku 1992, Fant 1985) were
  read on 2026-11-15; still unread: the D4C paper, and four known only from abstracts or
  snippets (see References). The Alku and Fant PDFs are scans; the Fant text is poor OCR.

## 6. Not proposed, and why

| Idea | Why not |
|---|---|
| WORLD breathy profile (aperiodicity, pitch-synchronous noise) | no consistent winner against the whisper mix; both judged artificial in several presentations |
| Klatt-style spectral tilt + H1 boost | heard as muffled and artificial |
| Kreiman's analysis-by-synthesis, Alku's inverse filtering, the LF model | Kreiman and Gerratt (2005) copy each voice by interactive inverse filtering and hand adjustment (cycle boundaries checked by the author); 20 listeners could not tell copies of 20 disordered voices from the originals, but voices with period doubling or biphonation were excluded and it is manual work per voice. Alku's PSIAIF (1992) is fully automatic but was shown only on synthetic vowels and healthy speakers, and fails for pressed phonation with a low first formant. Fant's LF model (1985, four parameters) describes glottal flow. None is a pipeline for thousands of automatically degraded utterances |
| Neural voice conversion (Jiao) | needs paired patient recordings, which are out of scope |
| AVQI as the severity scale | Maryn et al. (2010) define it on the central 3 s of a sustained /a/ concatenated with the voiced parts of the first two sentences of a phonetically balanced text, from six terms (CPPS; HNR, two shimmer measures, spectral slope and tilt), cut-off 2.95, validated on Dutch clinical voices against five raters (single-rater ICC 0.70, mean 0.92). The data here are digits and words, so it is not computable as defined, and its HNR and shimmer terms rest on cycle detection that fails on rough voices (Hillenbrand 1994) |

## 7. Decisions needed

Decided 2026-11-15 (nothing implemented yet):

1. **Where:** a `profile` key inside `degradation` (§ 3.1, § 3.3). The 128 enabled profiles (64 in the manifest
   + 64 smoke mirrors) got a new fingerprint; the old-to-new table of the 64 is
   [Thesis Profile Fingerprints](./Thesis-Profile-Fingerprints.md) (done 2026-11-15).
2. **Which profiles:** all five (`resynthesis_control`, `rough`, `tremor`, `voice_breaks`,
   `diplophonia`). Diplophonia was not told apart from jitter in listening and the aphonic
   gaps sounded defective: both rest on weak evidence. Levels: a provisional set of 17 is in
   § 3.4.1, awaiting the user's choice; each is calibrated in measured units (§ 3.4) before it
   goes into a profile.
3. **Perceptual validation:** none in the thesis, objective validation only (§ 4.2).
4. **Identity check:** a minimal pilot before the grid, with a cost report and explicit
   confirmation (§ 4.3).
5. **Wording of the whisper-mix limitation:** built from the literature, not from the
   author's own listening. Draft (English, adjust to the thesis language; the three papers
   cited were read in full: Englert and Kreiman on 2026-11-15 from the PDFs, Lopes earlier on PMC):

   > The noise-mixing degradation covers breathy and aphonic voice only, and its level is a
   > power fraction, not a clinical severity unit. It was not perceptually validated. Where a
   > simulated dysphonia has been validated by listening, it took trained listener panels
   > (Englert et al., 2017: 170 analysed listeners, who nevertheless took 36.4 % of real
   > dysphonic voices for synthetic ones) or a single expert rater with repeated items (Lopes et
   > al., 2017); neither was available here. Jitter and shimmer are not useful as independent
   > indices of perceived vocal quality, apart from their contribution to the overall pattern
   > of spectrally shaped noise (Kreiman and Gerratt, 2005), so the profiles of Section X are
   > described by the acoustic quantities they produce, not by a perceived severity.

Decided 2026-11-15, after the calibration (§ 3.4.2) and the identity pilot (§ 4.3):

6. **The inconclusive pilot:** a test with power (the per-utterance verification margin, § 4.4),
   its rule fixed and committed before it runs. Not chosen: accepting a non-neutral control,
   dropping the WORLD profiles, more repeats of the same pilot (the 1/4 term of the correction
   does not shrink with repeats).
7. **The 2 dB shimmer levels** of `rough` (jitter 2, 4, 7 %) become **1.5 dB** measured (request
   ≈1.7), readable by Praat's standard settings, so every level is described by a measured value.
8. **Diplophonia:** out of the thesis. It stays in the code as "not validated" and enters no
   grid; chapter 8 says so. (Investigating the S10 loss and removing it from the code were not
   chosen.)

## References

- Morise, Yokomori, Ozawa (2016) WORLD, IEICE Trans. Inf. Syst. E99-D(7), doi:10.1587/transinf.2015EDP7457; code: github.com/mmorise/World (modified BSD).
- Morise (2016) D4C, Speech Communication 84:57–65 (cited by the README; **not read**).
- Klatt, Klatt (1990) J. Acoust. Soc. Am. 87(2):820–857, doi:10.1121/1.398894.
- Kreiman, Antoñanzas-Barroso, Gerratt (2010) Behav. Res. Methods 42(4):1030–1041, doi:10.3758/BRM.42.4.1030.
- Skowronski, Shrivastav, Hunter (2015) J. Voice 29(6):670–681, doi:10.1016/j.jvoice.2014.11.005.
- Tayebi Arasteh et al. (2023) Sci. Rep. 13:20476, doi:10.1038/s41598-023-47711-7.
- Jiao, Tu, Berisha, Liss (2018) arXiv:1804.10325.
- Lopes, Freitas, Almeida, Silva, Alves (2017) Braz J Otorhinolaryngol 84(4):460–472, doi:10.1016/j.bjorl.2017.05.012 (read in full on PMC).
- Read in full on 2026-11-15 (PDFs in `documentation/03-articlesToRead/forTesis/medicine/`): Kreiman, Gerratt (2005) Perception of aperiodicity in pathological voice, J. Acoust. Soc. Am. 117(4):2201–2211, doi:10.1121/1.1858351; Englert, Madazio, Gielow, Lucero, Behlau (2017) Perceptual Error Analysis of Human and Synthesized Voices, J. Voice 31(4):516.e5–516.e18, doi:10.1016/j.jvoice.2016.12.015; Hillenbrand, Cleveland, Erickson (1994) JSHR 37(4):769–778, doi:10.1044/jshr.3704.769; Hillenbrand, Houde (1996) JSHR 39(2):311–321, doi:10.1044/jshr.3902.311; Maryn, De Bodt, Roy (2010) J. Commun. Disord. 43(3):161–174, doi:10.1016/j.jcomdis.2009.12.004; Alku (1992) Speech Commun. 11(2–3):109–118, doi:10.1016/0167-6393(92)90005-R (scan); Fant, Liljencrants, Lin (1985) A four-parameter model of glottal flow, STL-QPSR 26(4):1–13, KTH (a report, no DOI; scan, poor OCR).
- Not read; abstract or search snippet only (verify before citing): "Imitating dysphonic voice", Logoped Phoniatr Vocol 2019, doi:10.1080/14015439.2019.1659410; Gerratt, Kreiman (2001) analysis-by-synthesis, doi:10.1121/1.1409969; Shrivastav et al. (2012) J Speech Lang Hear Res 55(5):1407–1422, doi:10.1044/1092-4388(2012/11-0160); Englert et al. (2016) J. Voice 30(5):639.e17–e23, doi:10.1016/j.jvoice.2015.07.017 (the earlier study with fewer listeners that Englert 2017 builds on). Not used: Kreiman, Gerratt, Gabelman (2002) J. Acoust. Soc. Am. 112:2446, doi:10.1121/1.4780067, a one-page meeting abstract, not a paper.
