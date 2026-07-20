#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "models/lstm/LSTMAutoencoder.hpp"
#include "tensor/Tensor.hpp"

namespace guayaquil
{

using Tensor = nn::Tensor;

auto mse_between(const Tensor& a, const Tensor& b) -> float;
auto mae_between(const Tensor& a, const Tensor& b) -> float;

void compute_precision_recall_f1(const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    float& precision,
    float& recall,
    float& f1);

auto estimate_lstm_macs(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> std::size_t;
auto estimate_snn_macs(std::size_t input_features, int hidden_size, int layers) -> std::size_t;

template <typename T>
auto parameter_count(std::span<T*> params) -> std::size_t
{
    std::size_t count = 0;
    for (T* p : params)
    {
        if (!p) continue;
        count += static_cast<std::size_t>(p->size());
    }
    return count;
}

} // namespace guayaquil
