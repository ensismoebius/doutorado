# Statistics and Metrics — Plain Language Guide

> **Technical reference:** [Statistics](../Statistics.md)

---

## How do we know if the model is any good?

After training, you need numbers that summarise how well the system is performing. These are called *metrics*. Different metrics answer different questions.

---

## Classification metrics

For speaker verification, the model classifies each input as "this is speaker X" vs. "this is not speaker X" (or more generally, assigns to one of N speakers).

### Accuracy

Percentage of predictions that were correct:

```
accuracy = (number correct) / (total predictions)
```

Simple and intuitive. Problem: misleading when classes are imbalanced. If 95% of test samples are Speaker A, a model that always guesses A has 95% accuracy but is completely useless for the other speakers.

### Confusion matrix

A grid showing what the model predicted vs. what was actually true:

```
             Predicted A   Predicted B   Predicted C
Actual A        45              3             2
Actual B         4             41             5
Actual C         2              2            46
```

The diagonal = correct predictions. Off-diagonal = mistakes. Each cell shows *which* speakers are confused for which.

### Precision and Recall

For each class (e.g., "is this Speaker A?"):

**Precision**: of all the times the model said "this is A", how many were actually A?
```
precision = true positives / (true positives + false positives)
```
High precision → few false alarms.

**Recall**: of all the actual A samples, how many did the model correctly identify?
```
recall = true positives / (true positives + false negatives)
```
High recall → few misses.

**F1 score**: harmonic mean of precision and recall — gives one balanced number.

For speaker verification, recall (miss rate) and precision (false alarm rate) are both important for security:
- Low recall → legitimate users are rejected (inconvenience)
- Low precision → impostors are accepted (security breach)

---

## Regression metrics

When the model outputs a continuous number (like the paraconsistent D_truth score, or a reconstruction error):

**MSE** (Mean Squared Error): average of (predicted − true)². Penalises large errors heavily.

**MAE** (Mean Absolute Error): average of |predicted − true|. More robust to outliers.

**R²** (coefficient of determination): how much of the variance in the true values is explained by the model. R² = 1 is perfect; R² = 0 means the model is no better than predicting the mean; R² < 0 means the model is worse than predicting the mean.

---

## Running mean: tracking loss during training

The training loop calls `loss.update(batch_loss)` after each batch. At the end of the epoch, `loss.value()` returns the average loss over all batches. This is simpler than storing all batch losses and averaging.

---

## Cross-validation metrics

With k-fold cross-validation, each fold produces a separate metric (e.g., accuracy on fold 1 = 87%, fold 2 = 91%, fold 3 = 89%, ...). The reported metric is the average across folds, and optionally the standard deviation (which tells you how stable the results are).

Large standard deviation → the model is sensitive to which samples are in the training set → potentially unreliable.

---

## See also

- [Statistics (technical)](../Statistics.md) — metric formulas and API
- [K-Fold Cross-Validation (plain)](../../Concepts/Plain/K-Fold-Cross-Validation.md) — how to evaluate without bias
- [Paraconsistent (plain)](./Paraconsistent.md) — a pre-classification quality metric specific to this project
