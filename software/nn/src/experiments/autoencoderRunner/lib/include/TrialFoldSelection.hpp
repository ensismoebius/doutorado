/**
 * @file src/experiments/autoencoderRunner/lib/include/TrialFoldSelection.hpp
 * @brief TrialFoldSelection struct (extracted from TrialFoldSelector.hpp).
 */

#pragma once

#include <vector>

namespace autoencoderRunner
{

struct TrialFoldSelection
{
    std::vector<int> train_trial_ids;
    std::vector<int> val_trial_ids;
    std::vector<int> test_trial_ids; // Held-out test set (same across all folds)
};

} // namespace autoencoderRunner
