#ifndef NN_EXPERIMENTS_02_EXPERIMENT02TRAINING_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02TRAINING_HPP

#include <vector>

#include "Experiment02Evaluation.hpp"

auto k_fold_cross_validation(const std::vector<std::vector<double>>& features,
                             const std::vector<int>& labels, int k_folds, int random_seed)
    -> std::vector<FoldResult>;

#endif // NN_EXPERIMENTS_02_EXPERIMENT02TRAINING_HPP
