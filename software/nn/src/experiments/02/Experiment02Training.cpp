#include "Experiment02Training.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

#include "nn/layers/CrossEntropyLoss.hpp"
#include "nn/layers/SimpleResNet.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/statistics/multi_class_metrics.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/utility/batching.hpp"

using SNNResNet = SimpleResNet;

auto k_fold_cross_validation(const std::vector<std::vector<double>>& features,
                             const std::vector<int>& labels, int k_folds, int random_seed)
    -> std::vector<FoldResult>
{
    auto fold_function = [&](const std::vector<std::vector<double>>& train_features,
                             const std::vector<int>& train_labels,
                             const std::vector<std::vector<double>>& test_features,
                             const std::vector<int>& test_labels) -> FoldResult
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        ParaconsistentMetrics para_metrics =
            compute_paraconsistent_metrics(train_features, train_labels);

        int n_classes = 0;
        for (int label : labels) n_classes = std::max(n_classes, label + 1);

        SNNResNet model(train_features[0].size(), 128, n_classes);
        Adam optimizer(0.001F);
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

            nn::Tensor y(1, n_classes);
            y.at(0, train_labels[i]) = 1.0F;
            train_targets.emplace_back(y);
        }

        for (int epoch = 0; epoch < 10; ++epoch)
        {
            auto batches = create_batches(train_inputs, train_targets, 32);
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
            for (int c = 1; c < n_classes; ++c)
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
