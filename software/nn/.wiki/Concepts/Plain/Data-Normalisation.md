# Data Normalisation — Plain Language Guide

> **Technical reference:** [Data Normalisation](../Data-Normalisation.md)

---

## What is normalisation and why does it matter?

Raw signals come in many different scales. An EEG signal might range from −50 to +50 microvolts. A speech energy feature might range from 0.001 to 10,000. If you feed both into the same neural network without adjusting them, the network struggles — a small change in the large-valued feature dominates the gradient while the small-valued feature barely registers.

Normalisation puts all features on a comparable scale so the network can learn from all of them equally well.

Think of it like converting currencies: you can't add dollars and yen meaningfully without converting them first.

---

## Two common methods

### Z-score (standardisation)

Transforms each feature to have mean 0 and standard deviation 1:

```
normalised = (value − mean) / standard_deviation
```

After this transform, 68% of values fall in the range [−1, +1] and 95% in [−2, +2], regardless of the original scale.

**When to use:** When the data is roughly bell-shaped (Gaussian) and outliers matter.

### Min-max scaling

Squeezes values into [0, 1]:

```
normalised = (value − minimum) / (maximum − minimum)
```

**When to use:** When you want strict [0, 1] bounds — required by the paraconsistent α/β analysis in this project, because both coefficients are defined over unit-range values.

---

## Per-feature vs per-sample

These are two different things:

**Per-feature (column-wise):** Compute mean/std for each feature across all training samples, then apply. This is what you do for a standard classifier — make sure each input dimension is on a similar scale.

**Per-sample (row-wise):** Compute mean/std for each individual sample independently. This is common in signal processing where each recording might have a different baseline or amplitude level. In this project, EEG windows are z-scored per window, not globally.

---

## Important rule: fit on training data only

Normalisation statistics (mean, standard deviation, min, max) must be computed **only from the training set**, then applied to validation and test sets using those same statistics.

If you compute them from the whole dataset (including test), you introduce *data leakage* — the model indirectly "sees" test data during training. This makes results look better than they really are.

---

## Normalisation in this project

| Signal | Method | Scope |
|---|---|---|
| EEG windows | Z-score | Per window (sample-wise) |
| Audio features | Z-score | Per feature (column-wise, fit on training) |
| Paraconsistent input | Min-max [0, 1] | Required by the α/β algorithm |

---

## See also

- [Data Normalisation (technical)](../Data-Normalisation.md) — implementation details
- [K-Fold Cross-Validation (plain)](./K-Fold-Cross-Validation.md) — how to correctly apply normalisation with cross-validation
- [LFCC (plain)](./LFCC.md) — the features being normalised
