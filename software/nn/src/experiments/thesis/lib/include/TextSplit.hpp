/**
 * @file src/experiments/thesis/lib/include/TextSplit.hpp
 * @brief TextSplit struct (extracted from ThesisDataset.hpp).
 */

#pragma once

#include <cstddef>
#include <vector>

namespace thesis
{

// Split samples into text-dependent or text-independent train/test indices.
// text-dependent:   train and test use the same phrase set (split by utterance).
// text-independent: train and test use disjoint phrase sets.
struct TextSplit
{
    std::vector<size_t> train_indices;
    std::vector<size_t> test_indices;
};

} // namespace thesis
