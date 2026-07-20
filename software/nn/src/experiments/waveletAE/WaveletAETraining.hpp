/**
 * @file src/experiments/waveletAE/WaveletAETraining.hpp
 * @brief WaveletAEtraining.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_EXPERIMENTS_02_EXPERIMENT02TRAINING_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02TRAINING_HPP

#include <vector>

#include "WaveletAEEvaluation.hpp"

auto k_fold_cross_validation(const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    int k_folds,
    int random_seed) -> std::vector<FoldResult>;

#endif // NN_EXPERIMENTS_02_EXPERIMENT02TRAINING_HPP
