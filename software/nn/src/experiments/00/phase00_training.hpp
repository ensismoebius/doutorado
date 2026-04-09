/**
 * @file phase00_training.hpp
 * @brief Training utilities for PHASE 0 (tensorization, one-hot, training loop, saving).
 */

#ifndef EXPERIMENTS_00_PHASE00_TRAINING_HPP
#define EXPERIMENTS_00_PHASE00_TRAINING_HPP

#include <filesystem>
#include <memory>
#include <vector>

#include "Config.hpp"
#include "nn/layers/SimpleResNet.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

namespace phase00
{
struct TrainResult
{
    double accuracy;
    std::unique_ptr<SimpleResNet> model;
    int input_dim;
    int output_dim;
    int hidden_dim;
    int depth;
};

auto tensor_from_slice(const std::vector<std::vector<double>>& features, size_t start, size_t end)
    -> nn::Tensor;
auto one_hot_from_slice(const std::vector<int>& labels, size_t start, size_t end, int num_classes)
    -> nn::Tensor;
auto compute_accuracy(const nn::Tensor& logits, const std::vector<int>& labels) -> double;

auto train_resnet_snn(const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    const Config& cfg,
    int num_classes) -> TrainResult;

auto save_results(const std::filesystem::path& path,
    double alpha,
    double beta,
    double g1,
    double g2,
    double accuracy) -> void;
auto save_torch_state(const std::filesystem::path& path, const TrainResult& trained) -> void;

} // namespace phase00

#endif // EXPERIMENTS_00_PHASE00_TRAINING_HPP
