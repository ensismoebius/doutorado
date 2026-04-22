#include "../include/ComparativeTraining.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "../include/ComparativeEncoding.hpp"
#include "../include/ComparativeEvaluation.hpp"
#include "../include/ComparativeMetrics.hpp"
#include "nn/layers/losses/MSELoss.hpp"

namespace comparative_autoencoder_experiment
{

auto make_lstm_cfg(const ComparativeConfig& cfg) -> nn::models::lstm::LSTMAutoencoderConfig
{
    nn::models::lstm::LSTMAutoencoderConfig arch;
    arch.input_size = 1;
    arch.seq_len = cfg.window_size;
    arch.hidden_size = cfg.hidden_size;
    arch.latent_size = cfg.latent_size;
    arch.num_layers = 1;
    return arch;
}

auto make_snn_cfg(const ComparativeConfig& cfg, int layers, float alpha, float v_th)
    -> AutoencoderConfig
{
    AutoencoderConfig model_cfg;
    model_cfg.loss_type = "mse";
    model_cfg.input_features = cfg.window_size;
    model_cfg.hidden_size = cfg.hidden_size;
    model_cfg.latent_size = cfg.latent_size;
    model_cfg.depth = std::max(1, layers);
    model_cfg.time_step = 1.0f;
    model_cfg.resistance = 1.0f / std::max(v_th, 1e-3f);
    model_cfg.capacitance = std::max(1e-3f, -1.0f / std::log(std::max(alpha, 1e-3f)));
    model_cfg.encoder_layer_spec = {"linear:hidden:leaky", "linear:latent:identity"};
    model_cfg.decoder_layer_spec = {"linear:hidden:leaky", "linear:output:identity"};
    return model_cfg;
}

static void train_lstm_once(nn::models::lstm::LSTMAutoencoder& model,
    Adam& optimizer,
    const std::vector<Tensor>& train_samples,
    const std::string& encoding,
    std::uint32_t seed)
{
    MSELossImpl<nn::EigenTensorBackend> mse_loss;
    for (std::size_t i = 0; i < train_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(train_samples[i], encoding, seed + static_cast<std::uint32_t>(i));

        model.reset_state();
        optimizer.zero_grad(model.params());

        const Tensor recon = model.forward(encoded, true);
        mse_loss.set_target(encoded);
        const Tensor loss = mse_loss.forward(recon, true);
        (void) loss;
        const Tensor grad = mse_loss.backward(recon);
        model.backward(grad);
        optimizer.step(model.params());
    }
}

static void train_snn_once(ProtocolSpikingAutoencoder& model,
    Adam& optimizer,
    const std::vector<Tensor>& train_samples,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed)
{
    MSELossImpl<nn::EigenTensorBackend> mse_loss;
    for (std::size_t i = 0; i < train_samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(train_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);

        const Tensor flat = flatten_time_series(encoded);
        model.reset_state();
        optimizer.zero_grad(model.params());

        const Tensor recon_flat = model.forward(flat, true);
        mse_loss.set_target(flat);
        const Tensor loss = mse_loss.forward(recon_flat, true);
        (void) loss;
        const Tensor grad = mse_loss.backward(recon_flat);
        model.backward(grad);
        optimizer.step(model.params());
    }
}

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    Adam& optimizer,
    const ComparativeConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    float& train_ms,
    float& infer_ms) -> RunMetrics
{
    auto best = std::numeric_limits<float>::infinity();
    int bad_epochs = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        train_lstm_once(
            model, optimizer, train_samples, encoding, seed + static_cast<std::uint32_t>(epoch));

        float val_mse = 0.0f;
        MSELossImpl<nn::EigenTensorBackend> mse_loss;
        for (std::size_t i = 0; i < val_samples.size(); ++i)
        {
            const Tensor encoded =
                encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
            model.reset_state();
            const Tensor recon = model.forward(encoded, false);
            mse_loss.set_target(encoded);
            val_mse += mse_loss.forward(recon, false).at(0, 0);
        }
        if (!val_samples.empty()) val_mse /= static_cast<float>(val_samples.size());

        if (val_mse + 1e-8f < best)
        {
            best = val_mse;
            bad_epochs = 0;
        }
        else
        {
            ++bad_epochs;
            if (bad_epochs >= cfg.early_stop_patience) break;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        model.reset_state();
        (void) model.forward(encoded, false);
    }
    const auto infer_end = std::chrono::steady_clock::now();
    infer_ms = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();

    return evaluate_lstm(model,
        val_samples,
        std::vector<int>(val_samples.size(), 0),
        cfg.anomaly_tau,
        estimate_lstm_macs(make_lstm_cfg(cfg)),
        parameter_count(model.params()),
        encoding,
        seed,
        infer_ms);
}

auto train_with_early_stopping_snn(ProtocolSpikingAutoencoder& model,
    Adam& optimizer,
    const ComparativeConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    const std::string& encoding,
    const std::string& architecture,
    int layers,
    float alpha,
    float v_th,
    std::uint32_t seed,
    float& train_ms,
    float& infer_ms) -> RunMetrics
{
    auto best = std::numeric_limits<float>::infinity();
    int bad_epochs = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        train_snn_once(model,
            optimizer,
            train_samples,
            encoding,
            architecture,
            alpha,
            v_th,
            seed + static_cast<std::uint32_t>(epoch));

        float val_mse = 0.0f;
        MSELossImpl<nn::EigenTensorBackend> mse_loss;
        for (std::size_t i = 0; i < val_samples.size(); ++i)
        {
            Tensor encoded =
                encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
            encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
            const Tensor flat = flatten_time_series(encoded);
            model.reset_state();
            const Tensor recon = model.forward(flat, false);
            mse_loss.set_target(flat);
            val_mse += mse_loss.forward(recon, false).at(0, 0);
        }
        if (!val_samples.empty()) val_mse /= static_cast<float>(val_samples.size());

        if (val_mse + 1e-8f < best)
        {
            best = val_mse;
            bad_epochs = 0;
        }
        else
        {
            ++bad_epochs;
            if (bad_epochs >= cfg.early_stop_patience) break;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
        const Tensor flat = flatten_time_series(encoded);
        model.reset_state();
        (void) model.forward(flat, false);
    }
    const auto infer_end = std::chrono::steady_clock::now();
    infer_ms = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();

    return evaluate_snn(model,
        val_samples,
        val_labels,
        cfg.anomaly_tau,
        estimate_snn_macs(static_cast<std::size_t>(cfg.window_size), cfg.hidden_size, layers),
        parameter_count(model.params()),
        encoding,
        architecture,
        alpha,
        v_th,
        seed,
        infer_ms);
}

} // namespace comparative_autoencoder_experiment
