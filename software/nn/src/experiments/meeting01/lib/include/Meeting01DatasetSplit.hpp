#pragma once

#include <string>
#include <vector>

#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"
#include "tensor/Tensor.hpp"

namespace meeting01
{

using Tensor = nn::Tensor;
using WindowMetadata = nn::dataLoaders::fsdd::WindowMetadata;

// One nested leave-one-speaker-out fold.
//
// train / val / test hold disjoint sets of windows partitioned *by speaker*: no
// speaker and no source recording appears in more than one of the three. The SNN
// hyperparameter sweep selects on `val` only; the winning config is retrained on
// train ∪ val and evaluated once on `test`. Fixed-architecture baselines early-stop
// on `val` and are likewise evaluated once on `test`.
//
// *_meta is parallel to the corresponding *_samples vector and carries the
// speaker / recording / window identity needed for the split manifest and for
// hierarchical (recording-level, speaker-level) statistics downstream.
struct DatasetSplit
{
    std::vector<Tensor> train_samples;
    std::vector<Tensor> val_samples;
    std::vector<Tensor> test_samples;

    std::vector<WindowMetadata> train_meta;
    std::vector<WindowMetadata> val_meta;
    std::vector<WindowMetadata> test_meta;

    // Digit labels, kept for the existing evaluators' anomaly-scoring path.
    std::vector<int> val_labels;
    std::vector<int> test_labels;

    // Speaker names for this fold (for the manifest / logging).
    std::vector<std::string> train_speakers;
    std::string val_speaker;
    std::string test_speaker;
};

} // namespace meeting01
