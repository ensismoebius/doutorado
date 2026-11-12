/**
 * @file src/experiments/thesis/lib/include/TextSplit.hpp
 * @brief TextSplit struct (extracted from ThesisDataset.hpp).
 */

#pragma once

#include <cstddef>
#include <vector>

namespace thesis
{

/**
 * Which samples may train and which may be tested, by phrase ("seen" vs "unseen" phrases).
 *
 * The 11 phrases are put in a seeded order and cut into halves of equal size (5 + 5; the
 * 11th is used by neither mode):
 *
 *   mode          train phrases   test phrases
 *   -----------   -------------   --------------------------
 *   dependent     the first 5     the SAME 5    (phrases seen in training)
 *   independent   the first 5     the OTHER 5   (phrases never seen in training)
 *
 * The training set is identical in both modes, and the two test sets have the same number
 * of phrases, so the modes differ in one thing only: whether the test phrases were seen.
 * Train and test still never share a sample: the folds are speaker-disjoint, so the two
 * index lists are intersected with different speakers.
 *
 * Until 2026-11-12, "dependent" was a random 80/20 cut of every sample and "independent" a
 * 5/6 cut of the phrases: dependent trained on 80% of the data and independent on ~45%,
 * so any difference between the modes could be the training size.
 */
struct TextSplit
{
    std::vector<size_t> train_indices;
    std::vector<size_t> test_indices;
};

} // namespace thesis
