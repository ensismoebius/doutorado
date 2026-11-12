/**
 * @file src/experiments/thesis/lib/include/ParaconsistentSubsets.hpp
 * @brief The Phase 00 score of a feature set, recomputed on subsets of the speakers.
 *
 * Why. Phase 00 picks the extractor with alpha and beta computed over ALL 15 speakers, and
 * Phase 01 then tests that extractor on those same speakers. So the choice has seen the
 * test speakers. And the extractors are deterministic, so the three "repetitions" of a
 * handcrafted profile are one measurement: its sigma_d of 0 says nothing about how stable
 * the ranking is. Two families of subsets answer both:
 *
 *   subset           speakers kept                          answers
 *   --------------   ------------------------------------   ------------------------------
 *   loso:<s>         all but speaker s (15 subsets)         how much one speaker moves it
 *   fold:r<r>f<f>    the TRAINING speakers of Phase 01      would the choice change if it
 *                    repetition r, fold f (k x repeats)     had never seen the test fold?
 *
 * The folds are rebuilt exactly as Phase 01 builds them: GroupKFoldPolicy(k, shuffle, seed
 * + r) over the speaker of each sample. The features are not re-extracted per subset; for a
 * fixed extractor (handcrafted, PCA) that is exact, and for an autoencoder it means the
 * network was still trained on every speaker, a limit the output names.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ThesisFeatureExtraction.hpp"
#include "ThesisSample.hpp"

namespace thesis
{

struct SubsetScore
{
    std::string subset; ///< "loso:<subject_id>" | "fold:r<rep>f<fold>"
    std::string label;  ///< feature set label
    double alpha = 0.0;
    double beta = 0.0;
    double d_truth = 0.0;
    double d_penalized = 0.0;
};

/// Every feature set scored on the 15 leave-one-speaker-out subsets and on the training
/// speakers of every Phase 01 fold (fold_k folds, repetitions fold_seed + 0..fold_repeats-1).
auto score_speaker_subsets(const std::vector<ThesisSample>& samples,
    const std::vector<FeatureSet>& feature_sets,
    int fold_k,
    std::uint32_t fold_seed,
    int fold_repeats) -> std::vector<SubsetScore>;

/// <results_dir>/e05_<run_tag>_paraconsistent_subsets.csv
void write_subsets_csv(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<SubsetScore>& scores);

} // namespace thesis
