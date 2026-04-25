#include "../include/ComparativeTraining.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "../include/ComparativeEncoding.hpp"
#include "../include/ComparativeEvaluation.hpp"
#include "../include/ComparativeMetrics.hpp"
#include "nn/layers/losses/MSELoss.hpp"
#include "nn/utility/progress.hpp"

namespace comparative_autoencoder_experiment
{

auto extract_layer_sizes(const std::vector<std::string>& specs) -> std::vector<int>
{
    std::vector<int> sizes;
    for (const auto& spec : specs)
    {
        std::stringstream ss(spec);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ':'))
        {
            parts.push_back(token);
        }
        if (parts.size() >= 2 && parts[0] == "linear")
        {
            try { sizes.push_back(std::stoi(parts[1])); } catch (...) {}
        }
    }
    return sizes;
}

auto extract_latent_size(const std::vector<std::string>& encoder_specs, 
                         const std::vector<std::string>& decoder_specs) -> int
{
    auto get_size = [](const std::string& spec) -> int {
        std::stringstream ss(spec);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ':')) parts.push_back(token);
        if (parts.size() >= 2) {
            try { return std::stoi(parts[1]); } catch (...) {}
        }
        return -1;
    };

    if (!encoder_specs.empty()) {
        int s = get_size(encoder_specs.back());
        if (s != -1) return s;
    }
    if (!decoder_specs.empty()) {
        int s = get_size(decoder_specs.front());
        if (s != -1) return s;
    }
    return 16; // Final default
}

auto make_lstm_cfg(const ComparativeConfig& cfg) -> nn::models::lstm::LSTMAutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    nn::models::lstm::LSTMAutoencoderConfig arch;
    arch.input_size = 1;
    arch.seq_len = cfg.dataset.window_size;
    arch.hidden_size = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec) : sizes.front();
    arch.latent_size = extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);
    arch.num_layers = static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    return arch;
}

auto make_snn_cfg(const ComparativeConfig& cfg, float alpha, float v_th) -> AutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    const int effective_layers = static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    const int hidden_sz = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec) : sizes.front();

    AutoencoderConfig model_cfg;
    model_cfg.loss_type = "mse";
    model_cfg.input_features = cfg.dataset.window_size;
    model_cfg.hidden_size = hidden_sz;
    model_cfg.latent_size = extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);
    model_cfg.depth = effective_layers;
    model_cfg.layer_sizes = sizes;
    model_cfg.branch_hidden_size = cfg.model.branch_hidden_size;
    model_cfg.fusion_hidden_size = cfg.model.fusion_hidden_size;
    model_cfg.time_step = 1.0f;
    model_cfg.resistance = 1.0f / std::max(v_th, 1e-3f);
    model_cfg.capacitance = std::max(1e-3f, -1.0f / std::log(std::max(alpha, 1e-3f)));

    if (!cfg.model.encoder_layer_spec.empty())
    {
        model_cfg.encoder_layer_spec = cfg.model.encoder_layer_spec;
    }
    else
    {
        model_cfg.encoder_layer_spec = {"linear:hidden:leaky", "linear:latent:identity"};
    }

    if (!cfg.model.decoder_layer_spec.empty())
    {
        model_cfg.decoder_layer_spec = cfg.model.decoder_layer_spec;
    }
    else
    {
        model_cfg.decoder_layer_spec = {"linear:hidden:leaky", "linear:output:identity"};
    }

    model_cfg.branch_encoder_layer_spec = cfg.model.branch_encoder_layer_spec;
    model_cfg.branch_decoder_layer_spec = cfg.model.branch_decoder_layer_spec;
    model_cfg.fusion_encoder_layer_spec = cfg.model.fusion_encoder_layer_spec;
    model_cfg.fusion_decoder_layer_spec = cfg.model.fusion_decoder_layer_spec;
    return model_cfg;
}

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    Adam& optimizer,
    const ComparativeConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> RunMetrics
{
    auto best = std::numeric_limits<float>::infinity();
    int bad_epochs = 0;
    const std::size_t batch_sz = static_cast<std::size_t>(cfg.training.samples_per_batch);
    const std::size_t effective_batch_sz = (batch_sz == 0) ? 1 : batch_sz;
    const std::size_t total_batches =
        (train_samples.size() + effective_batch_sz - 1) / effective_batch_sz;
    const std::size_t total_training_samples =
        train_samples.size() * static_cast<std::size_t>(cfg.training.epochs);
    std::size_t seen_batches = 0;
    std::size_t processed_samples = 0;
    std::size_t completed_epochs = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < cfg.training.epochs; ++epoch)
    {
        MSELossImpl<nn::EigenTensorBackend> mse_loss;
        std::size_t batch_start = 0;
        while (batch_start < train_samples.size())
        {
            std::size_t batch_end = std::min(batch_start + effective_batch_sz, train_samples.size());
            optimizer.zero_grad(model.params());
            float epoch_loss = 0.0f;

            for (std::size_t i = batch_start; i < batch_end; ++i)
            {
                const Tensor encoded = encode_sample(
                    train_samples[i], encoding, seed + static_cast<std::uint32_t>(epoch + i));

                model.reset_state();
                const Tensor recon = model.forward(encoded, true);
                mse_loss.set_target(encoded);
                const Tensor loss_value = mse_loss.forward(recon, true);
                epoch_loss += loss_value.at(0, 0);

                const Tensor grad = mse_loss.backward(recon);
                model.backward(grad);
            }

            optimizer.step(model.params());
            seen_batches = static_cast<std::size_t>(epoch) * total_batches + (batch_start / effective_batch_sz) + 1;
            processed_samples = batch_end;

            const double avg_loss = epoch_loss / static_cast<float>(batch_end - batch_start);
printProgress(total_training_samples,
                effective_batch_sz,
                total_batches * cfg.training.epochs,
                seen_batches,
                processed_samples,
                false,
                run_id,
                total_runs,
                static_cast<std::size_t>(epoch + 1),
                static_cast<std::size_t>(cfg.training.epochs),
                batch_end,
                train_samples.size(),
                avg_loss,
                std::span<nn::Tensor*>{},
                "LSTM");

            batch_start = batch_end;
        }

        float val_mse = 0.0f;
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

        completed_epochs = static_cast<std::size_t>(epoch + 1);

        if (val_mse + 1e-8f < best)
        {
            best = val_mse;
            bad_epochs = 0;
        }
        else
        {
            ++bad_epochs;
            if (bad_epochs >= cfg.training.early_stop_patience) break;
        }
    }

printProgress(total_training_samples,
                effective_batch_sz,
                total_batches * cfg.training.epochs,
                seen_batches,
                processed_samples,
                true,
                run_id,
                total_runs,
                completed_epochs,
                static_cast<std::size_t>(cfg.training.epochs),
                train_samples.size(),
                train_samples.size(),
                best,
                std::span<nn::Tensor*>{},
                "LSTM");

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
         cfg.training.max_reconstruct_mean_deviation,
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
    float alpha,
    float v_th,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> RunMetrics
{
    auto best = std::numeric_limits<float>::infinity();
    int bad_epochs = 0;
    const std::size_t batch_sz = static_cast<std::size_t>(cfg.training.samples_per_batch);
    const std::size_t effective_batch_sz = (batch_sz == 0) ? 1 : batch_sz;
    const std::size_t total_batches =
        (train_samples.size() + effective_batch_sz - 1) / effective_batch_sz;
    const std::size_t total_training_samples =
        train_samples.size() * static_cast<std::size_t>(cfg.training.epochs);
    std::size_t seen_batches = 0;
    std::size_t processed_samples = 0;
    std::size_t completed_epochs = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < cfg.training.epochs; ++epoch)
    {
        MSELossImpl<nn::EigenTensorBackend> mse_loss;
        std::size_t batch_start = 0;
        while (batch_start < train_samples.size())
        {
            std::size_t batch_end = std::min(batch_start + effective_batch_sz, train_samples.size());
            optimizer.zero_grad(model.params());
            float epoch_loss = 0.0f;

            for (std::size_t i = batch_start; i < batch_end; ++i)
            {
                Tensor encoded = encode_sample(
                    train_samples[i], encoding, seed + static_cast<std::uint32_t>(epoch + i));
                encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);

                const Tensor flat = flatten_time_series(encoded);
                model.reset_state();
                const Tensor recon_flat = model.forward(flat, true);
                mse_loss.set_target(flat);
                const Tensor loss_value = mse_loss.forward(recon_flat, true);
                epoch_loss += loss_value.at(0, 0);

                const Tensor grad = mse_loss.backward(recon_flat);
                model.backward(grad);
            }

            optimizer.step(model.params());
            seen_batches = static_cast<std::size_t>(epoch) * total_batches + (batch_start / effective_batch_sz) + 1;
            processed_samples = batch_end;

            const double avg_loss = epoch_loss / static_cast<float>(batch_end - batch_start);
            printProgress(total_training_samples,
                effective_batch_sz,
                total_batches * cfg.training.epochs,
                seen_batches,
                processed_samples,
                false,
                run_id,
                total_runs,
                static_cast<std::size_t>(epoch + 1),
                static_cast<std::size_t>(cfg.training.epochs),
                batch_end,
                train_samples.size(),
                avg_loss,
                std::span<nn::Tensor*>{},
                "SNN");

            batch_start = batch_end;
        }

        float val_mse = 0.0f;
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

        completed_epochs = static_cast<std::size_t>(epoch + 1);

        if (val_mse + 1e-8f < best)
        {
            best = val_mse;
            bad_epochs = 0;
        }
        else
        {
            ++bad_epochs;
            if (bad_epochs >= cfg.training.early_stop_patience) break;
        }
    }

printProgress(total_training_samples,
                effective_batch_sz,
                total_batches * cfg.training.epochs,
                seen_batches,
                processed_samples,
                true,
                run_id,
                total_runs,
                completed_epochs,
                static_cast<std::size_t>(cfg.training.epochs),
                train_samples.size(),
                train_samples.size(),
                best,
                std::span<nn::Tensor*>{},
                "SNN");

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
         cfg.training.max_reconstruct_mean_deviation,
        estimate_snn_macs(static_cast<std::size_t>(cfg.dataset.window_size),
            extract_layer_sizes(cfg.model.encoder_layer_spec).empty() ? extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec) : extract_layer_sizes(cfg.model.encoder_layer_spec).front(),
            static_cast<int>(extract_layer_sizes(cfg.model.encoder_layer_spec).size())),
        parameter_count(model.params()),
        encoding,
        architecture,
        alpha,
        v_th,
        seed,
        infer_ms);
}

} // namespace comparative_autoencoder_experiment
