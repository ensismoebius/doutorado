/**
 * @file phase00_training.cpp
 * @brief Training loop and artifact writing for PHASE 0.
 *
 * This file keeps training logic explicit (PyTorch-like):
 * - build model
 * - optimizer attach
 * - epoch/batch loop with forward/loss/backward/step
 *
 * It also writes a small “torch-state-like” artifact for reproducibility.
 */

#include "phase00_training.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "nn/layers/Layers.hpp"
#include "nn/optimizers/Adam.hpp"

namespace phase00
{

auto tensor_from_slice(const std::vector<std::vector<double>>& features, size_t start, size_t end)
    -> nn::Tensor
{
    const size_t rows = end - start;
    const size_t cols = features.front().size();
    nn::Tensor result(rows, cols);

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result.at(i, j) = static_cast<float>(features[start + i][j]);
        }
    }

    return result;
}

auto one_hot_from_slice(const std::vector<int>& labels, size_t start, size_t end, int num_classes)
    -> nn::Tensor
{
    const size_t rows = end - start;
    nn::Tensor result(rows, static_cast<size_t>(num_classes));
    result.set_zero();

    for (size_t i = 0; i < rows; ++i)
    {
        result.at(i, labels[start + i]) = 1.0F;
    }

    return result;
}

auto compute_accuracy(const nn::Tensor& logits, const std::vector<int>& labels) -> double
{
    const auto samples = std::min<size_t>(logits.rows(), labels.size());
    if (samples == 0)
    {
        return 0.0;
    }

    size_t correct = 0;
    for (size_t i = 0; i < samples; ++i)
    {
        float max_val = logits.at(i, 0);
        int max_idx = 0;
        for (int j = 1; j < static_cast<int>(logits.cols()); ++j)
        {
            if (logits.at(i, j) > max_val)
            {
                max_val = logits.at(i, j);
                max_idx = j;
            }
        }
        if (max_idx == labels[i])
        {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(samples);
}

auto train_resnet_snn(const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    const Config& cfg,
    int num_classes) -> TrainResult
{
    const int input_dim = static_cast<int>(features.front().size());
    auto model = std::make_unique<SimpleResNet>(
        input_dim, cfg.resnet_hidden_dim, num_classes, cfg.resnet_depth);

    Adam optimizer(cfg.learning_rate);
    auto params = model->params();
    optimizer.attach(params);

    MSELoss loss;

    const int total = static_cast<int>(features.size());
    const int batch_size = std::min(cfg.batch_size, total);

    for (int epoch = 0; epoch < cfg.max_epochs; ++epoch)
    {
        for (int start = 0; start < total; start += batch_size)
        {
            const int end = std::min(start + batch_size, total);

            auto batch_x =
                tensor_from_slice(features, static_cast<size_t>(start), static_cast<size_t>(end));
            auto batch_y = one_hot_from_slice(
                labels, static_cast<size_t>(start), static_cast<size_t>(end), num_classes);

            loss.set_target(batch_y);
            optimizer.zero_grad(params);

            auto preds = model->forward(batch_x, true);
            (void) loss.forward(preds, true);
            auto grad = loss.backward(preds);
            model->backward(grad);
            optimizer.step(params);
        }
    }

    auto full_input = tensor_from_slice(features, 0, static_cast<size_t>(features.size()));
    auto final_logits = model->forward(full_input, false);
    double accuracy = compute_accuracy(final_logits, labels);

    return {accuracy,
        std::move(model),
        input_dim,
        num_classes,
        cfg.resnet_hidden_dim,
        cfg.resnet_depth};
}

auto save_results(const std::filesystem::path& path,
    double alpha,
    double beta,
    double g1,
    double g2,
    double accuracy) -> void
{
    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("Unable to open metrics file for writing");
    }

    file << "alpha,beta,g1,g2,accuracy\n";
    file << alpha << "," << beta << "," << g1 << "," << g2 << "," << accuracy << "\n";
}

auto save_torch_state(const std::filesystem::path& path, const TrainResult& trained) -> void
{
    nlohmann::json root;
    root["phase"] = 0;
    root["architecture"] = {
        {"type", "SimpleResNet"},
        {"input_dim", trained.input_dim},
        {"hidden_dim", trained.hidden_dim},
        {"output_dim", trained.output_dim},
        {"depth", trained.depth},
    };

    nlohmann::json state_dict = nlohmann::json::array();
    auto params = trained.model->params();
    for (size_t idx = 0; idx < static_cast<size_t>(params.size()); ++idx)
    {
        const auto* param = params[idx];
        nlohmann::json param_node;
        param_node["name"] = "param_" + std::to_string(idx);
        param_node["shape"] = {static_cast<int>(param->rows()), static_cast<int>(param->cols())};

        std::vector<float> flat;
        flat.reserve(param->size());
        for (size_t r = 0; r < param->rows(); ++r)
        {
            for (size_t c = 0; c < param->cols(); ++c)
            {
                flat.push_back(param->at(r, c));
            }
        }

        param_node["data"] = flat;
        state_dict.push_back(param_node);
    }

    root["state_dict"] = state_dict;

    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Unable to open torch state file for writing");
    }
    out << root.dump(2);
}

} // namespace phase00
