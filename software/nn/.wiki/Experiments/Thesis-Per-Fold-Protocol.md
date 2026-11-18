# Thesis: extractor and hyperparameters chosen inside each fold (pre-registration draft)

Status: **design agreed with the author 2026-11-18; nothing implemented, nothing run.**
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

## 6. Still open (to be fixed here before the code)

- Implementation route: per-fold Phase 00 inside the C++ binary (one process sees the split it
  scores), or the split exported and the ranking done in Python from stored features. Either way
  the fold assignment must be the one `GroupKFoldPolicy` produces, checked by a test.
- Cost estimate from one timed debug profile per kind (handcrafted, PCA, AE, DSNN search) on
  GridUnesp, run through the expensive-experiment guard with the author's confirmation.
