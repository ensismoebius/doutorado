#include "../include/E04Evaluation.hpp"

#include <algorithm>
#include <cmath>

#include "../include/E04Encoding.hpp"
#include "../include/E04Metrics.hpp"

namespace e04
{

using LstmTensor = nn::models::lstm::LSTMAutoencoder::Tensor;
using SnnTensor = ProtocolSpikingAutoencoder::Tensor;

auto evaluate_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float max_reconstruct_mean_deviation,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    std::uint32_t seed,
    float infer_ms) -> RunMetrics
{
    RunMetrics m;
    m.macs = macs;
    m.parameter_count = param_count;
    m.infer_ms = infer_ms;

    std::vector<int> pred_labels;
    pred_labels.reserve(val_samples.size());

    float mse_acc = 0.0f;
    float mae_acc = 0.0f;
    float y_mean_acc = 0.0f;
    std::size_t n_values = 0;

    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        model.reset_state();
        const Tensor recon = Tensor(model.forward(LstmTensor(encoded), false));

        mse_acc += mse_between(encoded, recon);
        mae_acc += mae_between(encoded, recon);

        float sample_residual_mean = 0.0f;
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            sample_residual_mean += std::fabs(encoded.at(k) - recon.at(k));
            y_mean_acc += encoded.at(k);
            ++n_values;
        }
        sample_residual_mean /= static_cast<float>(std::max<nn::Index>(1, encoded.size()));
        pred_labels.push_back(sample_residual_mean > max_reconstruct_mean_deviation ? 1 : 0);
    }

    m.mse = val_samples.empty() ? 0.0f : mse_acc / static_cast<float>(val_samples.size());
    m.mae = val_samples.empty() ? 0.0f : mae_acc / static_cast<float>(val_samples.size());

    float ss_tot = 0.0f;
    float ss_res = 0.0f;
    const float y_mean = (n_values > 0) ? y_mean_acc / static_cast<float>(n_values) : 0.0f;
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        model.reset_state();
        const Tensor recon = Tensor(model.forward(LstmTensor(encoded), false));
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            const float y = encoded.at(k);
            const float yh = recon.at(k);
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - y_mean) * (y - y_mean);
        }
    }
    m.r2 = (ss_tot > 1e-8f) ? (1.0f - (ss_res / ss_tot)) : 0.0f;

    compute_precision_recall_f1(val_labels, pred_labels, m.precision, m.recall, m.f1);

    m.spike_rate = 0.0f;
    m.energy = 10.0f * static_cast<float>(m.macs);

    return m;
}

auto evaluate_snn(ProtocolSpikingAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float max_reconstruct_mean_deviation,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed,
    float infer_ms) -> RunMetrics
{
    RunMetrics m;
    m.macs = macs;
    m.parameter_count = param_count;
    m.infer_ms = infer_ms;

    std::vector<int> pred_labels;
    pred_labels.reserve(val_samples.size());

    float mse_acc = 0.0f;
    float mae_acc = 0.0f;
    float y_mean_acc = 0.0f;
    std::size_t n_values = 0;
    float spike_sum = 0.0f;

    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);

        const Tensor flat = flatten_time_series(encoded);
        model.reset_state();
        const Tensor recon_flat = Tensor(model.forward(SnnTensor(flat), false));
        const Tensor recon = unflatten_time_series(recon_flat, encoded.rows(), encoded.cols());

        mse_acc += mse_between(encoded, recon);
        mae_acc += mae_between(encoded, recon);

        float sample_residual_mean = 0.0f;
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            sample_residual_mean += std::fabs(encoded.at(k) - recon.at(k));
            y_mean_acc += encoded.at(k);
            spike_sum += recon.at(k) > 0.0f ? 1.0f : 0.0f;
            ++n_values;
        }
        sample_residual_mean /= static_cast<float>(std::max<nn::Index>(1, encoded.size()));
        pred_labels.push_back(sample_residual_mean > max_reconstruct_mean_deviation ? 1 : 0);
    }

    m.mse = val_samples.empty() ? 0.0f : mse_acc / static_cast<float>(val_samples.size());
    m.mae = val_samples.empty() ? 0.0f : mae_acc / static_cast<float>(val_samples.size());

    float ss_tot = 0.0f;
    float ss_res = 0.0f;
    const float y_mean = (n_values > 0) ? y_mean_acc / static_cast<float>(n_values) : 0.0f;
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
        const Tensor flat = flatten_time_series(encoded);
        model.reset_state();
        const Tensor recon = unflatten_time_series(
            Tensor(model.forward(SnnTensor(flat), false)), encoded.rows(), encoded.cols());
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            const float y = encoded.at(k);
            const float yh = recon.at(k);
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - y_mean) * (y - y_mean);
        }
    }
    m.r2 = (ss_tot > 1e-8f) ? (1.0f - (ss_res / ss_tot)) : 0.0f;

    compute_precision_recall_f1(val_labels, pred_labels, m.precision, m.recall, m.f1);

    m.spike_rate = (n_values > 0) ? spike_sum / static_cast<float>(n_values) : 0.0f;
    m.energy = m.spike_rate * static_cast<float>(n_values) + 10.0f * static_cast<float>(m.macs);

    return m;
}

} // namespace e04
