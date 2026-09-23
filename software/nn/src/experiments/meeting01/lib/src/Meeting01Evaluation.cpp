#include "../include/Meeting01Evaluation.hpp"

#include <algorithm>
#include <cmath>

#include "../include/Meeting01Encoding.hpp"
#include "../include/Meeting01Metrics.hpp"

using nn::models::autoencoder::ProtocolSpikingAutoencoder;

namespace meeting01
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
    float infer_ms,
    int lstm_frame_size,
    int time_steps) -> RunMetrics
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
        // Compare in framed space: the model consumes and reconstructs frames, and
        // MSE/MAE/R2 are elementwise so framing both sides leaves them unchanged.
        const Tensor encoded = to_lstm_frames(
            encode_sample(
                val_samples[i], encoding, seed + static_cast<std::uint32_t>(i), time_steps),
            lstm_frame_size);
        const Tensor target =
            to_lstm_frames(make_reconstruction_target(val_samples[i], time_steps), lstm_frame_size);
        model.reset_state();
        const Tensor recon = Tensor(model.forward(LstmTensor(encoded), false));

        mse_acc += mse_between(target, recon);
        mae_acc += mae_between(target, recon);

        float sample_residual_mean = 0.0f;
        for (nn::Index k = 0; k < target.size(); ++k)
        {
            sample_residual_mean += std::fabs(target.at(k) - recon.at(k));
            y_mean_acc += target.at(k);
            ++n_values;
        }
        sample_residual_mean /= static_cast<float>(std::max<nn::Index>(1, target.size()));
        pred_labels.push_back(sample_residual_mean > max_reconstruct_mean_deviation ? 1 : 0);
    }

    m.mse = val_samples.empty() ? 0.0f : mse_acc / static_cast<float>(val_samples.size());
    m.mae = val_samples.empty() ? 0.0f : mae_acc / static_cast<float>(val_samples.size());

    float ss_tot = 0.0f;
    float ss_res = 0.0f;
    const float y_mean = (n_values > 0) ? y_mean_acc / static_cast<float>(n_values) : 0.0f;
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        // Compare in framed space: the model consumes and reconstructs frames, and
        // MSE/MAE/R2 are elementwise so framing both sides leaves them unchanged.
        const Tensor encoded = to_lstm_frames(
            encode_sample(
                val_samples[i], encoding, seed + static_cast<std::uint32_t>(i), time_steps),
            lstm_frame_size);
        const Tensor target =
            to_lstm_frames(make_reconstruction_target(val_samples[i], time_steps), lstm_frame_size);
        model.reset_state();
        const Tensor recon = Tensor(model.forward(LstmTensor(encoded), false));
        for (nn::Index k = 0; k < target.size(); ++k)
        {
            const float y = target.at(k);
            const float yh = recon.at(k);
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - y_mean) * (y - y_mean);
        }
    }
    m.r2 = (ss_tot > 1e-8f) ? (1.0f - (ss_res / ss_tot)) : 0.0f;

    binary_precision_recall_f1(val_labels, pred_labels, m.precision, m.recall, m.f1);

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
    float infer_ms,
    int time_steps) -> RunMetrics
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
        Tensor encoded = encode_sample(
            val_samples[i], encoding, seed + static_cast<std::uint32_t>(i), time_steps);
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
        const Tensor target = make_reconstruction_target(val_samples[i], time_steps);

        model.reset_state();
        const Tensor recon = Tensor(model.forward(SnnTensor(encoded), false));

        mse_acc += mse_between(target, recon);
        mae_acc += mae_between(target, recon);

        float sample_residual_mean = 0.0f;
        for (nn::Index k = 0; k < target.size(); ++k)
        {
            sample_residual_mean += std::fabs(target.at(k) - recon.at(k));
            y_mean_acc += target.at(k);
            spike_sum += recon.at(k) > 0.0f ? 1.0f : 0.0f;
            ++n_values;
        }
        sample_residual_mean /= static_cast<float>(std::max<nn::Index>(1, target.size()));
        pred_labels.push_back(sample_residual_mean > max_reconstruct_mean_deviation ? 1 : 0);
    }

    m.mse = val_samples.empty() ? 0.0f : mse_acc / static_cast<float>(val_samples.size());
    m.mae = val_samples.empty() ? 0.0f : mae_acc / static_cast<float>(val_samples.size());

    float ss_tot = 0.0f;
    float ss_res = 0.0f;
    const float y_mean = (n_values > 0) ? y_mean_acc / static_cast<float>(n_values) : 0.0f;
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        Tensor encoded = encode_sample(
            val_samples[i], encoding, seed + static_cast<std::uint32_t>(i), time_steps);
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
        const Tensor target = make_reconstruction_target(val_samples[i], time_steps);
        model.reset_state();
        const Tensor recon = Tensor(model.forward(SnnTensor(encoded), false));
        for (nn::Index k = 0; k < target.size(); ++k)
        {
            const float y = target.at(k);
            const float yh = recon.at(k);
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - y_mean) * (y - y_mean);
        }
    }
    m.r2 = (ss_tot > 1e-8f) ? (1.0f - (ss_res / ss_tot)) : 0.0f;

    binary_precision_recall_f1(val_labels, pred_labels, m.precision, m.recall, m.f1);

    m.spike_rate = (n_values > 0) ? spike_sum / static_cast<float>(n_values) : 0.0f;
    m.energy = m.spike_rate * static_cast<float>(n_values) + 10.0f * static_cast<float>(m.macs);

    return m;
}

auto per_window_errors_snn(ProtocolSpikingAutoencoder& model,
    const std::vector<Tensor>& samples,
    const std::vector<WindowMetadata>& meta,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed,
    int time_steps,
    PerWindowError proto) -> std::vector<PerWindowError>
{
    std::vector<PerWindowError> out;
    out.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(samples[i], encoding, seed + static_cast<std::uint32_t>(i), time_steps);
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
        const Tensor target = make_reconstruction_target(samples[i], time_steps);
        model.reset_state();
        const Tensor recon = Tensor(model.forward(SnnTensor(encoded), false));

        PerWindowError r = proto;
        if (i < meta.size())
        {
            r.speaker_id = meta[i].speaker_id;
            r.recording_id = meta[i].recording_id;
            r.window_id = meta[i].window_id;
            r.source_window_index = meta[i].source_window_index;
        }
        r.mse = mse_between(target, recon);
        r.mae = mae_between(target, recon);
        out.push_back(r);
    }
    return out;
}

} // namespace meeting01
