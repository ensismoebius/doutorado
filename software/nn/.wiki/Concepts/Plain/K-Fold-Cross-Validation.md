# K-Fold Cross-Validation — Plain Language Guide

> **Technical reference:** [K-Fold Cross-Validation](../K-Fold-Cross-Validation.md)

---

## The core problem

When you train a machine learning model, you need to know how well it will work on new data it has never seen. A simple but flawed approach: train on 80% of your data, test on 20%.

Problems:
- If you got lucky with the 80/20 split, results look great but are not representative.
- The 20% you evaluate on might not represent the full diversity of your dataset.
- You're "wasting" 20% of your data that the model never learns from.

---

## What k-fold cross-validation does

It solves this by repeating the train/test split k times, each time using a different portion as the test set:

```
Fold 1:  [TEST ] [train] [train] [train] [train]
Fold 2:  [train] [TEST ] [train] [train] [train]
Fold 3:  [train] [train] [TEST ] [train] [train]
Fold 4:  [train] [train] [train] [TEST ] [train]
Fold 5:  [train] [train] [train] [train] [TEST ]
```

Each fold trains a fresh model and evaluates it on the held-out slice. The final performance estimate is the average across all 5 folds.

This way: every sample gets tested exactly once, and you use all your data for both training and evaluation.

---

## How many folds? (the k=10 rule of thumb)

- **k=2 (50/50 split):** Each model trains on only half the data — high variance in results, unstable estimates.
- **k=5 or k=10:** Good balance. Each model trains on 80–90% of data. Most widely used.
- **k=n (leave-one-out):** Trains n separate models, each leaving out one sample. Very expensive for large datasets. Maximum use of training data but high compute cost.

Most papers use **k=5 or k=10**.

---

## Stratified k-fold

In speaker verification, classes may be imbalanced (more recordings from some speakers than others). Plain k-fold might put all recordings from one speaker in the same fold by accident.

**Stratified k-fold** ensures each fold has approximately the same proportion of each class. This is almost always the right choice for classification.

---

## Nested k-fold — avoiding a subtle cheat

Imagine you try 10 different learning rates and pick the one that gives the best cross-validation score. Now you report that score as your "unbiased" evaluation. But it is *not* unbiased — you peeked at the validation data when choosing the learning rate!

**Nested k-fold** separates hyperparameter tuning from performance estimation:

```
Outer loop (5 folds) ──→ reports unbiased test performance
  └── Inner loop (5 folds) ──→ selects best hyperparameters
```

For each outer fold:
1. Completely hide a test slice — never touch it.
2. Use the remaining data for inner cross-validation to find the best hyperparameters.
3. Train one final model with those hyperparameters.
4. Evaluate that model on the hidden test slice.

The outer loop scores are your honest estimate of how the system will perform on new data.

This matters a lot in small biomedical datasets like EEG recordings where overfitting to hyperparameter choices is a real risk.

---

## Data leakage — the most common mistake

**Data leakage** means your model indirectly sees test data during training. This inflates results and makes the evaluation worthless.

The most common source: **normalising before splitting**. If you compute mean/standard deviation across the entire dataset (including test folds) and then split, the normalisation statistics carry information from the test fold into the training process.

Correct order:
1. Split into folds.
2. Fit normalisation statistics on the training fold.
3. Apply those same statistics to the validation/test fold.
4. Never refit normalisation statistics on validation/test data.

---

## In this project

The library provides `KFold`, `StratifiedKFold`, and `NestedKFold` in `include/statistics/kfold.hpp`. The nested variant is the recommended default for all speaker authentication experiments to produce unbiased generalization estimates.

---

## See also

- [K-Fold Cross-Validation (technical)](../K-Fold-Cross-Validation.md) — implementation details and API
- [Data Normalisation (plain)](./Data-Normalisation.md) — how to normalise correctly inside folds
- [Autoencoders (plain)](./Autoencoders.md) — the models being evaluated
