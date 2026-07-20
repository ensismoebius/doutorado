/**
 * @file src/experiments/waveletAE/WaveletAETraining.cpp
 * @brief Implementation for WaveletAEtraining.
 *

 */

#include "WaveletAETraining.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

#include "layers/Layers.hpp"
#include "optimizers/Adam.hpp"
#include "statistics/multi_class_metrics.hpp"
#include "tensor/Tensor.hpp"
#include "utility/batching.hpp"

using SNNResNet = nn::SimpleResNet;
using nn::CrossEntropyLoss;

namespace
{
constexpr int kHiddenSize = 128;
constexpr int kTrainingEpochs = 10;
constexpr int kBatchSize = 32;
constexpr float kLearningRate = 0.001F;
} // namespace

auto k_fold_cross_validation(const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    int k_folds,
    int random_seed) -> std::vector<FoldResult>
{
    auto fold_function = [&](const std::vector<std::vector<double>>& train_features,
                             const std::vector<int>& train_labels,
                             const std::vector<std::vector<double>>& test_features,
                             const std::vector<int>& test_labels) -> FoldResult
    {
        if (train_features.empty() || train_labels.empty() || test_features.empty() ||
            test_labels.empty())
        {
            return FoldResult{};
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        ParaconsistentMetrics para_metrics =
            compute_paraconsistent_metrics(train_features, train_labels);

        const int class_count =
            labels.empty() ? 0 : (*std::max_element(labels.begin(), labels.end()) + 1);

        SNNResNet model(train_features[0].size(), kHiddenSize, class_count);
        Adam optimizer(kLearningRate);
        auto params = model.params();
        optimizer.attach(params);
        CrossEntropyLoss loss;

        std::vector<nn::Tensor> train_inputs;
        std::vector<nn::Tensor> train_targets;
        for (std::size_t i = 0; i < train_features.size(); ++i)
        {
            nn::Tensor x(1, static_cast<std::size_t>(train_features[i].size()));
            for (std::size_t j = 0; j < train_features[i].size(); ++j)
            {
                x.at(0, j) = train_features[i][j];
            }
            train_inputs.emplace_back(x);

            nn::Tensor y(1, class_count);
            y.setZero();
            y.at(0, train_labels[i]) = 1.0F;
            train_targets.emplace_back(y);
        }

        for (int epoch = 0; epoch < kTrainingEpochs; ++epoch)
        {
            // Seed the shuffle from the run seed, varied per epoch.
            //
            // Unseeded, create_batches() falls back to std::random_device, so batch order --
            // and therefore the order SGD sees -- differed on every run and the trained weights
            // with it. Offsetting by `epoch` keeps each epoch's order distinct (a fixed seed
            // would replay the same order every epoch, which defeats shuffling) while making
            // the whole schedule a pure function of `random_seed`.
            auto batches = create_batches(train_inputs,
                train_targets,
                kBatchSize,
                static_cast<unsigned int>(random_seed) + static_cast<unsigned int>(epoch));
            for (const auto& batch : batches)
            {
                loss.set_target(batch.targets);
                nn::Tensor logits = model.forward(batch.inputs);
                nn::Tensor grad_loss = loss.backward(logits);

                model.backward(grad_loss);
                optimizer.step(params);
            }
        }

        std::vector<int> pred_labels;
        for (const auto& test_feat : test_features)
        {
            nn::Tensor x(1, static_cast<std::size_t>(test_feat.size()));
            for (std::size_t j = 0; j < test_feat.size(); ++j)
            {
                x.at(0, j) = test_feat[j];
            }
            nn::Tensor output = model.forward(x);

            int pred = 0;
            float max_val = output.at(0, 0);
            for (int c = 1; c < class_count; ++c)
            {
                if (output.at(0, c) > max_val)
                {
                    max_val = output.at(0, c);
                    pred = c;
                }
            }
            pred_labels.push_back(pred);
        }

        ClassificationMetrics cls_metrics =
            compute_classification_metrics(test_labels, pred_labels);

        auto end_time = std::chrono::high_resolution_clock::now();
        double fold_time = std::chrono::duration<double>(end_time - start_time).count();

        return FoldResult{cls_metrics, para_metrics, fold_time};
    };

    return statistics::k_fold_cross_validation<FoldResult>(
        features, labels, k_folds, random_seed, fold_function);
}
