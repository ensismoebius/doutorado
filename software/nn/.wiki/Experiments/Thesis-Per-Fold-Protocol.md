# Thesis: extractor and hyperparameters chosen inside each fold (pre-registration draft)

Status: **design agreed with the author 2026-11-18; stages (a)-(f) of §6 implemented, tested
and run on `thesis-per-fold-protocol`.** Runs of record (deploy `271cbb5b`, one CPU per task):
`perfold` 300/300 units (job 5331068), `pfbiomex` 144/144 (5335100 + rerun 5338456 at 16G),
`pfworld` 150/150 (5345042), `pfbxworld` running (5345820, 208/270 on 2026-11-25). Tables and the
declared tests are generated for the first three (§6 amendments); the thesis text still quotes
the fixed-configuration results.
This page is the rule. It is committed before any code that produces a result, and a result
never changes it: a deviation is written here as a dated amendment, with its reason, before the
run it affects.

## 1. The problem

The thesis results (Phase 01) carry two choices made with information the test speakers helped
produce:

| Choice | How it is made today | Who sees the test speakers |
|---|---|---|
| Feature extractor (Phase 00 winner per signal) | $D_{\text{pen}}$ ranked on **all 15** speakers | the ranking does |
| PCA / autoencoder weights | fitted on **all 15** speakers (unsupervised) | the fit does |
| Classifier architecture and training | fixed by hand (`linear:128`, `residual:2`, lr 1e-3, no decay) | nobody, but nothing is tuned either |

Example of why it matters: if the voice winner is chosen because it separates the 15 speakers
best, then fold 3, which tests speakers 7, 11 and 14, is scored with an extractor those three
speakers helped elect. The thesis reports this as a limitation ("o extrator é escolhido com
todos os locutores"; the bias is nil on EEG, where the choice is the same in every fold, and
real on voice). A model that is never tuned, on the other hand, describes one configuration
and not what each family can reach ("os hiperparâmetros não foram ajustados").

Cost is no longer the obstacle: the slowest Phase 00 profile took 230 s on GridUnesp (array
5323006), and a DSNN fold trains in about 1.3 s (`train_ms` in the Phase 01 summaries).

## 2. The protocol

For every repetition (seed) and every outer fold of the existing speaker-disjoint split
(`GroupKFoldPolicy`, k = 5, 12 training speakers, 3 test speakers), using **only the 12
training speakers** until the single test evaluation:

```
outer fold f (seed s)
├── 12 train speakers ──┬─ Phase 00 inside the fold
│                       │    · handcrafted: rank the stored feature vectors by D_pen on these 12
│                       │    · PCA and autoencoders: fit on these 12, then rank by D_pen on these 12
│                       │    · winner per signal (voice, EEG pronounced, EEG imagined)
│                       ├─ hyperparameter search (inner GroupKFold, k = 4: 9 train / 3 validation speakers)
│                       │    · every candidate of §3, mean validation EER over the 4 inner folds
│                       │    · winner = lowest mean EER; ties -> fewer parameters
│                       └─ refit the winner on all 12
└── 3 test speakers ──── scored ONCE (EER, AUC)
```

- **Fused sources** use the fold's voice winner for both parts, as today.
- **DSNN and its non-spiking twin get the same search** (same candidates, same inner folds, same
  selection rule); the cosine control has no hyperparameters and uses the fold's extractor.
- **Comparison configurations** (the extractor comparison of Phase 01) choose, inside the fold,
  the best member of the family they declare.
- The selected extractor and hyperparameters of every (seed, fold) are written to the results,
  so the stability of the choices is reported, not hidden.

## 3. Hyperparameter search space (author's choice: compact, T fixed)

| Hyperparameter | Values |
|---|---|
| Width of the hidden layers | 64, 128, 256 |
| Residual blocks (after the input stage) | 0, 1, 2 |
| Learning rate (Adam) | 1e-3, 3e-4 |
| Weight decay (matrices only) | 0, 1e-4 |

36 candidates for each network. T = 16 for the DSNN in every candidate, so the spiking network
and its twin have the same number of candidates (a T axis for the DSNN alone would give it
three times the chances). Everything else stays as in the current profiles (50 epochs, batch 32,
no clipping, no firing-rate regularisation, no tdBN).

## 4. Scope (author's choices)

- **PCA and autoencoders are refitted inside every fold** on the training speakers only.
- **This protocol becomes the primary result.** The current Phase 01 numbers move to a
  comparison section: how much the speaker-informed choice and the missing tuning moved them.
  The limitation about the extractor chosen with all speakers leaves the thesis.
- **BIOMEX-DB uses the same protocol per fold**, with the same search space: the replication
  repeats the method, not the choices. The WORLD grids follow the same protocol.

## 5. Estimands and tests (unchanged from the current thesis)

The same comparisons, the same statistics as the current Phase 01 (the corrected t test and
Holm correction of the thesis, chapter 3) and the same CLAIMS mechanism: every number of
the prose comes from `thesis_build_text_numbers.py`. What changes is only how each test-fold
model was produced.

## 6. Implementation (amendment 2026-11-18: route chosen by the author — inside the binary)

One `thesis` process runs one (profile, repetition) and does everything of §2 itself, so the
split it ranks on is, by construction, the split it scores. Nothing crosses a file boundary.

**Profile.** A Phase 01 profile gains a `selection` block; when it is enabled, the keys it owns
are refused elsewhere in the profile (the strict-config rule: a declared value that changes
nothing is a silent fallback):

```json
"selection": {
  "enabled": true,
  "rank_modality": "voice",                 // the signal the candidates are ranked on
  "extractor_candidates": [ {feature_extraction}, ... ],   // the Phase 00 set of that signal
  "inner_k": 4,
  "hidden_widths": [64, 128, 256],          // dsnn / ann only
  "residual_blocks": [0, 1, 2],             //   layer_spec residual:<blocks + 1>
  "learning_rates": [0.001, 0.0003],
  "weight_decays": [0.0, 0.0001]
}
```

Owned (refused when `selection.enabled`): `feature_extraction`, `classifier.layer_spec`,
`training.learning_rate`, `training.weight_decay`, `training.nested_cv`. The cosine control
declares only the extractor part. `rank_modality` is the profile's own signal, except for the
fused sources, where it is `voice` (§2: the fold's voice winner feeds both parts).

**Per outer fold** (the existing `GroupKFoldPolicy(k, shuffle, seed)` split, intersected with the
text split exactly as `run_flat_cv` does):

1. Candidate features. Handcrafted: extracted once per process for every sample (a sample's
   vector does not depend on any other sample). PCA and autoencoders: fitted on the fold's
   training rows only, then applied to every sample; the autoencoder's training frames keep the
   sample's own index in `frame_seed`, so a training sample is encoded with the frames it was
   trained on.
2. $D_{\text{pen}}$ of every candidate on the fold's training rows (`score_feature_set`); winner =
   lowest; ties -> first in profile order (the generator writes them in the Phase 00 order).
3. Features of the profile's own modality with the winner (for fused: the winner applied as in
   `extract_features`, early or late).
4. Inner `GroupKFoldPolicy(inner_k, shuffle, seed ^ 0xDEADBEEF)` over the training rows' speakers;
   every candidate of the grid trained on inner-train (scaler fitted there) and scored by EER on
   inner-validation; winner = lowest mean EER; ties -> fewer trainable parameters, then grid
   order.
5. Refit the winner on all training rows (scaler fitted there), score the test rows once
   (degraded test rows, when the profile degrades, extracted with the fold's winner).
6. The fold record carries the winning extractor's label, its $D_{\text{pen}}$, the winning
   hyperparameters and the mean inner EER of every grid candidate.

**Which rows, exactly** (amendment 2026-11-18, stage (c): fixed in `ThesisPerFold.hpp`). The
text split $T$ of the repetition's seed (`make_text_split`) cuts every speaker's rows by phrase,
so "the 12 training speakers" is not yet a row list. The rule is the outer fold copied one level
down:

| Rows | Speakers | Phrases | Used for |
|---|---|---|---|
| training | the 12 training speakers | $T$.train | extractor fit, $D_{\text{pen}}$, refit |
| test | the 3 test speakers | $T$.test | the single evaluation |
| inner train | the 9 inner-train speakers | $T$.train | each grid candidate's model and scaler |
| inner validation | the 3 inner-validation speakers | $T$.test | each grid candidate's EER |

Example (text-independent mode): if $T$.train = {arriba, abajo, ...} and $T$.test = {adelante,
...}, an inner-validation speaker is scored on "adelante" exactly as a test speaker is, and no
model of the fold ever trains on "adelante". The extractor is never fitted on a $T$.test row,
of any speaker: that is the text leak fixed on 2026-11-12, one level up.

**Degraded test rows with a fitted winner** (PCA, autoencoder). The degraded features must come
from the SAME model as the clean ones. The run builds a view whose training rows keep the clean
voice and whose other rows carry the degraded one, fits the winner on the training rows (so the
fit is the clean fit: it reads only those rows, and every autoencoder frame is seeded by the
sample's own index) and compares the training rows of both extractions bit for bit; a
difference stops the run. Only test rows are read from the degraded extraction. A handcrafted
winner fits nothing: its degraded features are extracted once per process. This lifts, for the
per-fold protocol only, the refusal of degradation with an autoencoder.

**What a fold records** (`FoldResult::selection`): every candidate's label and $D_{\text{pen}}$
(profile order) and the winner's index; every grid point with its parameter count, its EER per
inner fold and their mean, and the winner's index; the refitted winner's parameter count;
`fold_ms`, the wall-clock of the whole fold (extraction, ranking, search, refit, test).
`train_ms` stays the refit's, so it compares with the fixed-configuration run. The run-level
`param_count` is absent (each fold's winner has its own). The result's label is
`per-fold-<modality>` (`per-fold-fused-<mode>` for fused).

**Outputs** (amendment 2026-11-18, stage (d)). Beside the usual metrics, comparison and
learning-curve files of the repetition:

| File / key | Content |
|---|---|
| summary `"protocol"` | `"per-fold"` (`"fixed"` in every other run) |
| summary `"strategy"` | absent under the per-fold protocol, so a reader that groups by extractor fails loudly instead of filing the run under one |
| summary `fold_models[f].selection` | `extractors` [{label, d_penalized}], `extractor_index`, `extractor_label`, `param_count`, `fold_ms`; networks also `grid` [{hidden_width, residual_blocks, learning_rate, weight_decay, param_count, inner_eer[], mean_inner_eer}] and `grid_index` (the cosine control has no grid keys) |
| `e05_<tag>_selection_extractors.csv` | `feature_set,fold,candidate,label,d_penalized,chosen`, one row per (fold, candidate) |
| `e05_<tag>_selection_grid.csv` | `feature_set,fold,point,hidden_width,residual_blocks,learning_rate,weight_decay,param_count,inner_fold,inner_eer,mean_inner_eer,chosen`, one row per (fold, point, inner fold); header only for the cosine control |

The Python table builders (`thesis_build_phase01_auth_tables.py` and the rest) read the
profile's `feature_extraction.strategy`, which a per-fold profile does not have: they must be
adapted before they read a per-fold run. (Amendment 2026-11-19, after the `perfold` array: the
Phase 01 builder takes `--protocol per-fold` and adds `perfold_selection.csv` and the
descriptive `perfold_vs_fixed.csv`; `10_thesis_perfold_significance.sh` runs the families of
`03_` on the `pf01_*` names, as §5 requires; the significance engine pairs a per-fold profile
with a flat fixed one, its outer folds being the flat CV's. BIOMEX-DB: amendment 2026-11-23, after `pfbiomex`: `thesis_build_biomex_tables.py --protocol per-fold` writes `pfbiomex_*.csv` beside the Coretto per-fold cell, and `11_thesis_pfbiomex_significance.sh` is `04_`'s families on the `pfbx01_*` names. Same renaming-only rule as `10_`. WORLD: amendment 2026-11-24, after `pfworld`: `08_thesis_world_analysis.py --protocol per-fold --bases ...` runs Part B (the EER of every level against its cell's control, the same rule) on `pfworld`/`pfbxworld` into `world_eer_perfold.csv`. Part A, the confirmatory margin, is not re-run: it reads `thesis_features`, one vector per utterance from one extractor per profile, with no classifier, so it has no per-fold version; it stays the fixed extractor's.)

**Profiles** (amendment 2026-11-18, stage (e); `00_thesis_generate_profiles.py`, `per_fold()`).
Every fixed-configuration profile now declares `"selection": {"enabled": false}` and stays as
the comparison grid. The per-fold profiles mirror every flat, standardized cell of it, in their
own directories (results in `results/thesis/<dir>`):

| Directory | Profiles | Count |
|---|---|---|
| `perfold/` | `pf01_<dsnn,ann,cosine>_<7 sources>_<dep,indep>`; `pf01_dysph_<5 voice sources>_<scope>_l<level>`; `pf01_cmp-<hc,ae,pca>_<3 signals>_<text>` | 42 + 40 + 18 |
| `pfbiomex/` | `pfbx01_<kind>_<4 sources>_<text>`; `pfbx01_dysph_<3 sources>_<scope>_l<level>` | 24 + 24 |
| `pfworld/`, `pfbxworld/` | `pf01w_…`, `pfbx01w_…`: the WORLD grid | 150 + 90 |

Not mirrored: the DSNN's nested-CV cells (the selection is what replaces them) and its raw
(unstandardized) cells (an ablation of the fixed grid); the identity pilot. Candidates: the
Phase 00 set of the ranked signal, in Phase 00 order (voice: 153 = 138 handcrafted + 12
autoencoders + 3 PCA; EEG: 61), each autoencoder with the 100-epoch block Phase 00 ranked it
at. The `cmp-<family>` profiles keep only that family and now train the classifier 50 epochs
like every other cell (the fixed grid needed 100 because one training block drove both). The
per-fold profiles read no winners, and a fusion ranks the voice whatever `--fused` says.

**Smoke check** (local, 2026-11-18, smoke mirror: 240 samples, 2 folds, autoencoders 2 epochs,
one grid point, 2 inner folds): `pf01_cosine_eegpron_dep` and `pf01_dysph_voice_test_l100`
ran end to end on the real Coretto data; in the second, fold 0 chose an ANN-AE and the
degraded refit check passed on it; 522 s on the workstation for the degraded voice profile.
The smoke mirror needs 240 samples, not 120: PCA k = 32 is fitted on a fold's training rows,
about a quarter of the samples there.

**Timed debug run on GridUnesp** (stage (f), 2026-11-18, deploy `74b80587`, one CPU per
task, the full protocol: no reduction). One unit per kind; its fold 0 includes the one-time
extraction of the ranked signal's handcrafted candidates.

| Unit | Wall-clock | Peak RSS | Fold 0 | Folds 1-4, each |
|---|---|---|---|---|
| `pf01_dsnn_voice_dep.rep0` | 24 981 s | 4.5 GB | 15 564 s | ~2 350 s |
| `pf01_dysph_voice_test_l100.rep0` | 22 450 s | 6.0 GB | 13 927 s | ~2 100 s |
| `pf01_ann_voice_dep.rep0` | 14 296 s | 4.5 GB | 12 750 s | ~390 s |
| `pf01_cosine_voice_dep.rep0` | 13 547 s | 4.5 GB | 12 590 s | ~240 s |
| `pf01_dsnn_eegpron_dep.rep0` | 10 380 s | 2.1 GB | 2 522 s | ~1 950 s |
| `pfbx01_dsnn_voice_dep.rep0` | 45 349 s | 5.1 GB | 15 922 s | ~7 400 s |
| `pf01w_voice_test_rough-j2` (3 repetitions) | ~16.8 h (projected from 13.5 h) | | | ~2 300 s |

Read from the differences: the voice's 138 handcrafted candidates take ~12 350 s per process
(Coretto; ~8 500 s on BIOMEX-DB), the DSNN grid ~2 100 s per fold (~150 s for the ANN), the
autoencoder/PCA fits plus the ranking ~240 s per fold. Projected without a cache: ~9 100 CPU·h
for the four phases, and a whole `pfbxworld` profile ~43 h, beyond the partition's day. The
dysphonia unit chose exactly what the clean DSNN unit chose in every fold (same clean training
rows, same seed), as it must.

**Feature cache** (amendment 2026-11-18, author's choice after the debug run;
`FeatureDiskCache.hpp`). The handcrafted candidates of the ranked signal are computed once per
data view by a warm array (`thesis_warm_feature_cache.sbatch`, `thesis --warm-feature-cache`;
11 views: Coretto voice, eegpron, eegimag and the 4 fusions, BIOMEX-DB voice, eegpron and the 2
pronounced fusions) and read by every unit of the four phases, which refuse a miss. Key:
deploy revision + fingerprint of every loaded value and label + signal + the candidate's whole
`feature_extraction` block; the file holds the key and a checksum, and a mismatch is refused.
Nothing else is cached (PCA and the autoencoders are fitted per fold; the winner's fused or
degraded features are a few extractions per fold). Verified: a per-fold run gives bit-identical
choices and EERs with and without the cache (`FeatureDiskCache.PerFoldRunIsTheSameWithAndWithoutIt`,
and locally on the real Coretto EEG smoke: EER 0.310185 both ways). `pfbxworld` is split by
repetition. The debug units' markers carry deploy `74b80587`: the phase directories must be
moved aside before the full run of a new deploy.

**Refused with `selection.enabled`**: `paraconsistent.enabled = true` — the whole-dataset ranking
needs one extractor for every sample, and here each fold has its own.

**Stages** (each with its tests, committed on `thesis-per-fold-protocol`):
(a) fit-on-rows for PCA and autoencoders; (b) the `selection` block in `ThesisConfig` with the
refusals; (c) the per-fold runner; (d) outputs; (e) the profile generator; (f) a timed debug
profile per kind on GridUnesp, only with the author's confirmation, before any real run.
