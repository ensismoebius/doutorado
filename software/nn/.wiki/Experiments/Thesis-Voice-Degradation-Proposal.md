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
hashes the declared keys (`thesis.cpp`), so the fingerprints of the 128 enabled profiles change
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

1. **Where:** a `profile` key inside `degradation` (§ 3.1, § 3.3). The 128 enabled profiles get
   a new fingerprint; an old-to-new table is needed.
2. **Which profiles:** all five (`resynthesis_control`, `rough`, `tremor`, `voice_breaks`,
   `diplophonia`). Diplophonia was not told apart from jitter in listening and the aphonic
   gaps sounded defective: both rest on weak evidence. Levels are still open; they are
   calibrated in measured units (§ 3.4) before they go into a profile.
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
