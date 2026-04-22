# K-Fold Cross-Validation

K-fold cross-validation is a robust technique for estimating model performance and hyperparameter tuning.

## Theoretical Background

### Standard K-Fold

The dataset is split into $k$ equal-sized folds:

1. Train on $k-1$ folds, validate on 1 fold
2. Repeat $k$ times, each fold used exactly once for validation
3. Average the $k$ validation scores

Final metric: $\bar{m} = \frac{1}{k} \sum_{i=1}^{k} m_i$

### Bias-Variance Tradeoff [6]

- **Low k** (e.g., 2): Low bias, high variance (few training examples per fold)
- **High k** (e.g., 10): High bias, low variance (more training examples)
- **k = n** (leave-one-out): Maximum variance, minimum bias

Research suggests **k=10** provides good balance for most problems.

### Stratified K-Fold

Maintains class distribution across folds:
- In classification: each fold has same proportion of each class
- Essential for imbalanced datasets

### With Test Split

Extended version with dedicated test set:
```
Train+Val: 80% → split into k folds for CV
Test: 20% → held out completely, evaluated once
```

## How It Is Implemented Here

### Fold Sampler

```cpp
// File: include/nn/dataLoaders/samplers/FoldSampler.hpp
class FoldSampler : public ISampler
{
    size_t total_size_;
    size_t num_folds_;
    size_t current_fold_;
    size_t seed_;

public:
    auto get_train_indices(size_t epoch) -> std::vector<size_t> override
    {
        // Returns indices excluding current validation fold
    }

    auto get_val_indices() -> std::vector<size_t> override
    {
        // Returns indices for current validation fold
    }
};
```

### K-Fold Configuration

```cpp
// File: include/nn/statistics/kfold.hpp
struct KFoldConfig
{
    int num_folds = 5;
    float test_split = 0.0f;  // 0.0 = no test set, 0.2 = 20% held out
    unsigned int seed = 42;
    bool shuffle = true;
};
```

## Data Flow

```mermaid
flowchart LR
    subgraph Data
        data[Full Dataset<br/>N samples]
    end

    subgraph Split
        test[Test Set<br/>test_split × N]
        cv[Train+Val<br/>(1-test_split) × N]
    end

    subgraph CV
        fold1[Fold 1: Val]
        fold2[Fold 2: Val]
        fold3[Fold 3: Val]
    end

    subgraph Results
        scores[m1, m2, ..., mk]
        avg[Average]
    end

    data --> test
    data --> cv
    cv --> fold1
    cv --> fold2
    cv --> fold3
    fold1 --> scores
    fold2 --> scores
    fold3 --> scores
    scores --> avg
```

## Usage Example

```cpp
// File: src/experiments/03/lib/src/TrialFoldSelector.cpp
#include "nn/statistics/kfold.hpp"
#include "nn/dataLoaders/samplers/FoldSampler.hpp"

// Configure k-fold
nn::statistics::KFoldConfig kfold_cfg{
    .num_folds = 5,
    .test_split = 0.2f,  // 20% test set
    .seed = 42
};

// Run k-fold training
std::vector<float> fold_scores;

for (int fold = 0; fold < kfold_cfg.num_folds; ++fold)
{
    // Get fold splits
    auto train_indices = get_train_indices(fold, kfold_cfg);
    auto val_indices = get_val_indices(fold, kfold_cfg);

    // Create data loaders
    DataLoader train_loader(dataset, batch_size, train_indices);
    DataLoader val_loader(dataset, batch_size, val_indices);

    // Train model
    auto model = create_model();
    Trainer trainer(*model, config);
    auto history = trainer.fit_autoencoder(train_loader, val_loader);

    // Record validation score
    fold_scores.push_back(history.back().val_loss);
}

// Report results
float mean_score = std::accumulate(fold_scores.begin(), fold_scores.end(), 0.0f) / fold_scores.size();
std::cout << "K-Fold Mean: " << mean_score << std::endl;
```

## Common Pitfalls

1. **Data Leakage**: Never use validation data for training; preprocess separately per fold

2. **Non-IID Data**: Time-series or grouped data require special handling

3. **Test Split**: Always hold out test set to estimate true generalization error

4. **Stratification**: Use stratified sampling for classification with imbalanced classes

## See Also

- [DataLoaders](../Core/DataLoaders.md) - Loading and sampling data
- [Autoencoders](./Autoencoders.md) - Models being validated
- [Experiment03](../Experiments/Experiment03.md) - Experiments using k-fold

## References

[1] R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in Proc. 14th Int. Joint Conf. Artificial Intelligence (IJCAI), 1995, pp. 1137–1143.
[2] S. Arlot and A. Celisse, "A survey of cross-validation procedures for model selection," *Statistics Surveys*, vol. 4, pp. 40–79, 2010. [Online]. Available: https://doi.org/10.1214/09-SS054