# DataLoaders

The nn library provides a flexible data loading system for batching, shuffling, and preprocessing datasets.

## Theoretical Background

Data loaders enable efficient training by:

1. **Batching**: Grouping samples into mini-batches for stochastic gradient descent
2. **Shuffling**: Randomizing order to prevent overfitting to sequence patterns
3. **Prefetching**: Loading next batch while GPU computes current batch
4. **Sampling Strategies**: Controlling which samples are included in each epoch

### K-Fold Cross-Validation

K-fold splits data into $k$ folds for robust model evaluation [6]:

```
Fold 1: [val] [train train train]
Fold 2: [train] [val train train]
Fold 3: [train train] [val train]
Fold 4: [train train train] [val]
```

## How It Is Implemented Here

```cpp
// File: include/nn/dataLoaders/runtime/DataLoader.hpp
class DataLoader
{
    std::shared_ptr<Dataset> dataset_;
    std::size_t batch_size_;
    std::unique_ptr<ISampler> sampler_;

public:
    using Iterator = DataLoaderIterator;

    auto begin() -> Iterator;
    auto end() -> Iterator;
};
```

### Samplers

```cpp
// File: include/nn/dataLoaders/samplers/ISampler.hpp
class ISampler
{
public:
    virtual auto get_indices(size_t epoch) -> std::vector<size_t> = 0;
};
```

Available samplers:
- `SequentialSampler` - No shuffling
- `RandomSampler` - Random shuffling with seed
- `FoldSampler` - K-fold cross-validation splits
- `DistributedSampler` - Multi-GPU training

### Dataset Interface

```cpp
// File: include/nn/dataLoaders/datasets/Dataset.hpp
class Dataset
{
public:
    virtual auto size() const -> size_t = 0;
    virtual auto get(size_t index) -> Tensor = 0;
    virtual auto collate(const std::vector<size_t>& indices) -> Tensor = 0;
};
```

## Data Flow

```mermaid
flowchart TB
    subgraph Data
        Raw[Raw Files<br/>.mat, .csv]
    end

    subgraph Loading
        DS[Dataset]
        SP[Sampler]
        DL[DataLoader]
    end

    subgraph Batching
        Collate[collate()]
        Prefetch[BatchPrefetcher]
    end

    subgraph Training
        Batch[Batch Tensor]
    end

    Raw --> DS
    DS --> SP
    SP --> DL
    DL --> Collate
    Collate --> Prefetch
    Prefetch --> Batch
```

## Usage Example

```cpp
// File: include/nn/dataLoaders/runtime/DataLoader.hpp
#include "nn/dataLoaders/runtime/DataLoader.hpp"
#include "nn/dataLoaders/datasets/MatFileDataset.hpp"

// Create dataset from MAT file
auto dataset = std::make_shared<nn::dataLoaders::MatFileDataset>("data.mat");

// Create data loader with random shuffling
DataLoader loader(dataset, batch_size=32, do_shuffle=true, seed=42);

// Iterate batches
for (const auto& batch : loader)
{
    // batch is a Tensor of shape (32, features)
    auto output = model.forward(batch, true);
    // ... training step
}
```

### K-Fold Example

```cpp
// File: include/nn/dataLoaders/samplers/FoldSampler.hpp
#include "nn/dataLoaders/samplers/FoldSampler.hpp"

// Create k-fold sampler
nn::dataLoaders::FoldSampler fold_sampler(
    dataset->size(),  // total samples
    5,                // number of folds
    fold_index,       // which fold is validation
    seed
);

// Train on fold
DataLoader train_loader(dataset, batch_size, fold_sampler.get_train_indices());
DataLoader val_loader(dataset, batch_size, fold_sampler.get_val_indices());
```

## Common Pitfalls

1. **Batch Size**: Too large causes poor generalization; too small causes slow training

2. **Seed**: Always set seed for reproducibility in experiments

3. **Prefetching**: Use `BatchPrefetcher` for GPU training to overlap I/O and compute

4. **Shuffle Between Epochs**: Always reshuffle between epochs to prevent overfitting

## See Also

- [Tensor](./Tensor.md) - Output tensor type
- [K-Fold Cross-Validation](../Concepts/K-Fold-Cross-Validation.md) - Cross-validation details
- [Data Normalisation](../Concepts/Data-Normalisation.md) - Input preprocessing

## References

[1] R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in Proc. 14th Int. Joint Conf. Artificial Intelligence (IJCAI), 1995, pp. 1137–1143.

[2] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
