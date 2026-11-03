#include "../include/Meeting01Evaluation.hpp"

#include <algorithm>
#include <cmath>

#include "../include/Meeting01Encoding.hpp"
#include "../include/Meeting01Energy.hpp"
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

    // No spiking neuron: spike_rate stays NaN (not applicable) and acs 0.
    m.arith_energy_nj = arith_energy_nj(static_cast<double>(m.macs), 0.0);

    return m;
}

auto evaluate_snn(ProtocolSpikingAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<WindowMetadata>& val_meta,
    const std::vector<int>& val_labels,
    float max_reconstruct_mean_deviation,
    std::size_t param_count,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed,
    float infer_ms,
    int time_steps,
    int output_steps) -> RunMetrics
{
    RunMetrics m;
    m.parameter_count = param_count;
    m.infer_ms = infer_ms;

    std::vector<int> pred_labels;
    pred_labels.reserve(val_samples.size());

    float mse_acc = 0.0f;
    float mae_acc = 0.0f;
    float y_mean_acc = 0.0f;
    std::size_t n_values = 0;
    SnnOpCounts ops;
    EnergyLedger energy; // summed over the windows, averaged below

    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor raw = encode_sample(
            val_samples[i], encoding, seed + static_cast<std::uint32_t>(i), time_steps);
        const Tensor encoded = apply_snn_architecture_transform(raw, architecture, alpha, v_th);
        const Tensor target = make_reconstruction_target(val_samples[i], output_steps);
        const Tensor mask = make_reconstruction_target(
            make_activity_mask(val_meta[i].valid_length, static_cast<int>(val_samples[i].size())),
            output_steps);

        model.reset_state();
        const SnnTensor input(encoded);
        const Tensor recon = Tensor(model.forward(input, false));
        const SnnOpCounts window_ops =
            count_snn_ops(model, input, snn_input_is_spikes(encoding, architecture));
        ops.macs += window_ops.macs + count_architecture_transform_macs(
                                          architecture, static_cast<std::size_t>(encoded.size()));
        ops.acs += window_ops.acs;
        ops.spikes += window_ops.spikes;
        ops.neuron_steps += window_ops.neuron_steps;

        // The memory-aware estimate: the encoder that made the spikes, the input transform,
        // then the network. Only the "dense" transform reads the encoder's output as events;
        // conv1d and recurrent read every (value, step), so the encoder writes all of them.
        const auto values = static_cast<double>(val_samples[i].size());
        energy += encoding_energy(encoding,
            values,
            time_steps,
            encoding == "direct" ? 0.0 : static_cast<double>(raw.sum()),
            /*event_driven=*/architecture == "dense");
        energy += architecture_transform_energy(architecture,
            values,
            time_steps,
            architecture == "recurrent" ? static_cast<double>(encoded.sum()) : 0.0);
        energy += snn_ae_energy(model, input, snn_input_is_spikes(encoding, architecture));

        mse_acc += mse_between_masked(target, recon, mask);
        mae_acc += mae_between_masked(target, recon, mask);

        float sample_residual_sum = 0.0f;
        float sample_mask_sum = 0.0f;
        for (nn::Index k = 0; k < target.size(); ++k)
        {
            const float mk = mask.at(k);
            sample_residual_sum += mk * std::fabs(target.at(k) - recon.at(k));
            sample_mask_sum += mk;
            if (mk > 0.0f)
            {
                y_mean_acc += target.at(k);
                ++n_values;
            }
        }
        const float sample_residual_mean =
            (sample_mask_sum > 0.0f) ? sample_residual_sum / sample_mask_sum : 0.0f;
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
        const Tensor target = make_reconstruction_target(val_samples[i], output_steps);
        const Tensor mask = make_reconstruction_target(
            make_activity_mask(val_meta[i].valid_length, static_cast<int>(val_samples[i].size())),
            output_steps);
        model.reset_state();
        const Tensor recon = Tensor(model.forward(SnnTensor(encoded), false));
        for (nn::Index k = 0; k < target.size(); ++k)
        {
            if (mask.at(k) <= 0.0f) continue;
            const float y = target.at(k);
            const float yh = recon.at(k);
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - y_mean) * (y - y_mean);
        }
    }
    m.r2 = (ss_tot > 1e-8f) ? (1.0f - (ss_res / ss_tot)) : 0.0f;

    binary_precision_recall_f1(val_labels, pred_labels, m.precision, m.recall, m.f1);

    // Per-window means of the measured counts. No windows, or a network with no spiking
    // layer, leaves spike_rate NaN: there is no fraction to report.
    if (!val_samples.empty())
    {
        const auto n = static_cast<double>(val_samples.size());
        m.macs = static_cast<std::size_t>(std::llround(ops.macs / n));
        m.acs = ops.acs / n;
        m.arith_energy_nj = arith_energy_nj(ops.macs / n, m.acs);
        set_energy(m, energy.scaled(1.0 / n));
    }
    if (ops.neuron_steps > 0.0) m.spike_rate = static_cast<float>(ops.spikes / ops.neuron_steps);

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
    int output_steps,
    PerWindowError proto) -> std::vector<PerWindowError>
{
    std::vector<PerWindowError> out;
    out.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(samples[i], encoding, seed + static_cast<std::uint32_t>(i), time_steps);
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
        const Tensor target = make_reconstruction_target(samples[i], output_steps);
        const Tensor mask = make_reconstruction_target(
            make_activity_mask(meta[i].valid_length, static_cast<int>(samples[i].size())),
            output_steps);
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
        r.mse = mse_between_masked(target, recon, mask);
        r.mae = mae_between_masked(target, recon, mask);
        out.push_back(r);
    }
    return out;
}

} // namespace meeting01
