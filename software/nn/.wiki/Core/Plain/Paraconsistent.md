# Paraconsistent Feature Engineering — Plain Language Guide

> **Technical reference:** [Paraconsistent Feature Engineering](../Paraconsistent.md)

---

## The core question this answers

After extracting feature vectors from speech or EEG signals, you have a pile of numbers. The question is: **are these features actually useful for telling speakers apart?**

You could just train a classifier and hope for the best. But that takes time, and if the features are bad, you wasted it. The paraconsistent method gives you a quality score *before* training any classifier.

---

## The intuition: clusters vs clouds

Imagine plotting your feature vectors for two speakers on a 2D map. If the features are good:
- Speaker A's points cluster tightly together
- Speaker B's points cluster tightly in a different region
- The two clusters don't overlap

If the features are bad:
- All points are scattered randomly everywhere
- You can't tell Speaker A from Speaker B by looking at the map

The paraconsistent analysis quantifies exactly this: how tight are the clusters (α) and how much do they overlap (β)?

---

## The two measurements

### α — Intraclass Similarity ("how tight are the clusters?")

For each speaker class, look at how much the feature values vary across their recordings:
- If every recording from Speaker A produces nearly identical feature vectors → tight cluster → high α
- If recordings from Speaker A are all over the place → loose cluster → low α

**α = 1** → perfect: every class forms a tight point in feature space  
**α = 0** → terrible: at least one class is scattered across the entire possible range

### β — Interclass Overlap ("do the clouds mix?")

For each pair of speakers, count how many of Speaker A's feature values happen to fall inside Speaker B's range (and vice versa):
- If A and B occupy completely different regions → no overlap → β = 0
- If A and B are completely mixed together → maximum overlap → β = 1

**β = 0** → perfect: the classes don't touch at all  
**β = 1** → terrible: the classes are completely interleaved

---

## The paraconsistent plane

Now map (α, β) to a 2D plane using:
- **G₁ = α − β** (certainty: high when features are both tight and separated)
- **G₂ = α + β − 1** (contradiction: high when features are tight but still overlapping)

The four corners of this plane tell a story:

| Corner | α | β | Meaning |
|---|---|---|---|
| **(1, 0) = Truth** | High | Low | Classes tight and separated → great features |
| **(-1, 0) = Falsity** | Low | High | Classes mixed everywhere → useless features |
| **(0, 1) = Ambiguity** | High | High | Tight but overlapping → confusingly similar |
| **(0, -1) = Indefinition** | Low | Low | Scattered but not mixing → incoherent data |

---

## The quality score: D_truth

The single number you care about is the **distance from (G₁, G₂) to the Truth corner (1, 0)**:

```
D_truth = sqrt((G₁ - 1)² + G₂²)
```

**Smaller D_truth = better features.**

D_truth = 0 → perfect separation (never happens in practice)  
D_truth ≈ 0.3 → strong features, simple classifiers will work  
D_truth > 0.8 → features carry little discriminative information

---

## How it's used in the thesis pipeline

The pipeline tries many combinations of (wavelet type × frequency scale):
- Haar wavelet + BARK scale
- Haar wavelet + MEL scale
- Haar wavelet + LFCC scale
- Daubechies-4 wavelet + BARK scale
- ... (all combinations)

For each combination, compute the feature vectors for all speakers, then compute D_truth. The combination with the **smallest D_truth** is selected as the best feature extraction strategy — without training a single classifier.

This replaces an expensive grid search over classifiers with a fast geometric quality score.

---

## What makes this novel?

The connection to da Costa's paraconsistent logic is the key innovation. Classical logic says something is either True or False. Paraconsistent logic allows states that are both/neither — capturing the "ambiguous" and "indefinite" quadrants in the plane above.

This framing naturally handles the case where features are "partially separable" — not clean enough to be easy, not hopeless — and gives a geometrically meaningful score.

---

## Important: normalise first

The algorithm requires all feature values to be in [0, 1] before computing α and β. Unnormalised features give meaningless results because α and β are based on range (max − min), which is meaningless if different dimensions have wildly different scales.

---

## See also

- [Paraconsistent Feature Engineering (technical)](../Paraconsistent.md) — formulas, API, mermaid diagram
- [LFCC (plain)](../../Concepts/Plain/LFCC.md) — one of the feature scales being evaluated
- [Wavelet (plain)](./Wavelet.md) — the feature extraction step before paraconsistent analysis
- [Research Context](../../Research-Context.md) — where this fits in the thesis
