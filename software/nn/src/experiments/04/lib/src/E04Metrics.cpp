#include "../include/ComparativeMetrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace comparative_autoencoder_experiment
{

auto mse_between(const Tensor& a, const Tensor& b) -> float
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        throw std::invalid_argument("mse_between shape mismatch");
    }
    float sum = 0.0f;
    const nn::Index n = a.size();
    for (nn::Index i = 0; i < n; ++i)
    {
        const float d = a.at(i) - b.at(i);
        sum += d * d;
    }
    return (n > 0) ? (sum / static_cast<float>(n)) : 0.0f;
}

auto mae_between(const Tensor& a, const Tensor& b) -> float
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        throw std::invalid_argument("mae_between shape mismatch");
    }
    float sum = 0.0f;
    const nn::Index n = a.size();
    for (nn::Index i = 0; i < n; ++i)
    {
        sum += std::fabs(a.at(i) - b.at(i));
    }
    return (n > 0) ? (sum / static_cast<float>(n)) : 0.0f;
}

void compute_precision_recall_f1(const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    float& precision,
    float& recall,
    float& f1)
{
    int tp = 0;
    int fp = 0;
    int fn = 0;
    for (std::size_t i = 0; i < y_true.size() && i < y_pred.size(); ++i)
    {
        if (y_true[i] == 1 && y_pred[i] == 1) ++tp;
        if (y_true[i] == 0 && y_pred[i] == 1) ++fp;
        if (y_true[i] == 1 && y_pred[i] == 0) ++fn;
    }

    precision = (tp + fp) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
    recall = (tp + fn) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
    f1 = (precision + recall) > 0.0f ? (2.0f * precision * recall) / (precision + recall) : 0.0f;
}

auto estimate_lstm_macs(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> std::size_t
{
    const std::size_t T = static_cast<std::size_t>(cfg.seq_len);
    const std::size_t I = static_cast<std::size_t>(cfg.input_size);
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t L = static_cast<std::size_t>(cfg.num_layers);

    const std::size_t per_gate = H * (I + H);
    const std::size_t per_step = 4 * per_gate;
    const std::size_t per_stack = per_step * L;
    const std::size_t proj = H * static_cast<std::size_t>(cfg.latent_size) +
                             static_cast<std::size_t>(cfg.latent_size) * H + H * I;
    return T * per_stack + proj;
}

auto estimate_snn_macs(std::size_t input_features, int hidden_size, int layers) -> std::size_t
{
    const std::size_t H = static_cast<std::size_t>(hidden_size);
    const std::size_t L = static_cast<std::size_t>(std::max(1, layers));
    const std::size_t in_proj = input_features * H;
    const std::size_t hidden_proj = (L > 1) ? (L - 1) * H * H : 0;
    const std::size_t out_proj = H * input_features;
    return in_proj + hidden_proj + out_proj;
}

} // namespace comparative_autoencoder_experiment
