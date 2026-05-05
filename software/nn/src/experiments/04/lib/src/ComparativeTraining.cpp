#include "../include/ComparativeTraining.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "../include/BatchLossCollector.hpp"
#include "../include/ComparativeEncoding.hpp"
#include "../include/ComparativeEvaluation.hpp"
#include "../include/ComparativeMetrics.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "nn/training/EarlyStoppingCallback.hpp"
#include "nn/training/ProgressCallback.hpp"

namespace comparative_autoencoder_experiment
{

using LstmTensor = nn::models::lstm::LSTMAutoencoder::Tensor;
using SnnTensor = ProtocolSpikingAutoencoder::Tensor;

// ---------------------------------------------------------------------------
// Config helpers (unchanged)
// ---------------------------------------------------------------------------

auto extract_layer_sizes(const std::vector<std::string>& specs) -> std::vector<int>
{
    std::vector<int> sizes;
    for (const auto& spec : specs)
    {
        std::stringstream ss(spec);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ':')) parts.push_back(token);
        if (parts.size() >= 2 && parts[0] == "linear")
        {
            try
            {
                sizes.push_back(std::stoi(parts[1]));
            }
            catch (...)
            {
            }
        }
    }
    return sizes;
}

auto extract_latent_size(const std::vector<std::string>& encoder_specs,
    const std::vector<std::string>& decoder_specs) -> int
{
    auto get_size = [](const std::string& spec) -> int
    {
        std::stringstream ss(spec);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ':')) parts.push_back(token);
        if (parts.size() >= 2)
        {
            try
            {
                return std::stoi(parts[1]);
            }
            catch (...)
            {
            }
        }
        return -1;
    };

    if (!encoder_specs.empty())
    {
        int s = get_size(encoder_specs.back());
        if (s != -1) return s;
    }
    if (!decoder_specs.empty())
    {
        int s = get_size(decoder_specs.front());
        if (s != -1) return s;
    }
    return 16;
}

auto make_lstm_cfg(const ComparativeConfig& cfg) -> nn::models::lstm::LSTMAutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    nn::models::lstm::LSTMAutoencoderConfig arch;
    arch.input_size = 1;
    arch.seq_len = cfg.dataset.window_size;
    arch.hidden_size = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                           cfg.model.decoder_layer_spec)
                                     : sizes.front();
    arch.latent_size =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);
    arch.num_layers = static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    return arch;
}

auto make_snn_cfg(const ComparativeConfig& cfg, float alpha, float v_th) -> AutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    const int effective_l = static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    const int hidden_sz = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                              cfg.model.decoder_layer_spec)
                                        : sizes.front();

    AutoencoderConfig model_cfg;
    model_cfg.loss_type = "mse";
    // After flatten_time_series, input is {1, window_size*1} — SNN sees window_size features.
    model_cfg.input_features = cfg.dataset.window_size;
    model_cfg.hidden_size = hidden_sz;
    model_cfg.latent_size =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);
    model_cfg.depth = effective_l;
    model_cfg.layer_sizes = sizes;
    model_cfg.branch_hidden_size = cfg.model.branch_hidden_size;
    model_cfg.fusion_hidden_size = cfg.model.fusion_hidden_size;
    model_cfg.time_step = 1.0f;
    model_cfg.resistance = 1.0f / std::max(v_th, 1e-3f);
    model_cfg.capacitance = std::max(1e-3f, -1.0f / std::log(std::max(alpha, 1e-3f)));

    model_cfg.encoder_layer_spec =
        cfg.model.encoder_layer_spec.empty()
            ? std::vector<std::string>{"linear:hidden:leaky", "linear:latent:identity"}
            : cfg.model.encoder_layer_spec;

    model_cfg.decoder_layer_spec =
        cfg.model.decoder_layer_spec.empty()
            ? std::vector<std::string>{"linear:hidden:leaky", "linear:output:identity"}
            : cfg.model.decoder_layer_spec;

    model_cfg.branch_encoder_layer_spec = cfg.model.branch_encoder_layer_spec;
    model_cfg.branch_decoder_layer_spec = cfg.model.branch_decoder_layer_spec;
    model_cfg.fusion_encoder_layer_spec = cfg.model.fusion_encoder_layer_spec;
    model_cfg.fusion_decoder_layer_spec = cfg.model.fusion_decoder_layer_spec;
    return model_cfg;
}

// ---------------------------------------------------------------------------
// Build a TrainerConfig from ComparativeConfig
// ---------------------------------------------------------------------------

static auto make_trainer_config(const ComparativeConfig& cfg, float snn_lr_scale = 1.0F)
    -> nn::training::TrainerConfig
{
    nn::training::TrainerConfig tcfg;
    tcfg.epochs = cfg.training.epochs;
    tcfg.batch_size = std::max(1, cfg.training.samples_per_batch);
    tcfg.learning_rate = cfg.training.learning_rate;
    tcfg.snn_lr_scale = snn_lr_scale;
    return tcfg;
}

// ---------------------------------------------------------------------------
// LSTM training
// ---------------------------------------------------------------------------

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    Adam& /*optimizer*/, // Trainer owns its optimizer; kept in signature for compatibility
    const ComparativeConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult
{
    // LSTM processes one sequence at a time (no 3-D batching).
    nn::training::TrainerConfig tcfg = make_trainer_config(cfg);
    tcfg.batch_size = 1;

    nn::training::Trainer<nn::models::lstm::LSTMAutoencoder> trainer(model, tcfg);

    const std::string label =
        "LSTM [" + encoding + "] r" + std::to_string(run_id + 1) + "/" + std::to_string(total_runs);
    trainer.add_callback(std::make_shared<nn::training::ProgressCallback>(label));

    auto stopper =
        std::make_shared<nn::training::EarlyStoppingCallback>(cfg.training.early_stop_patience);
    trainer.add_callback(stopper);

    auto batch_collector = std::make_shared<BatchLossCollector>();
    trainer.add_callback(batch_collector);

    std::vector<LstmTensor> train_backend_samples;
    train_backend_samples.reserve(train_samples.size());
    for (const auto& sample : train_samples) train_backend_samples.emplace_back(sample);

    std::vector<LstmTensor> val_backend_samples;
    val_backend_samples.reserve(val_samples.size());
    for (const auto& sample : val_samples) val_backend_samples.emplace_back(sample);

    // Reset LSTM state and encode each sample before forward.
    trainer.set_sample_transform(
        [&model, &encoding, seed](const LstmTensor& s, std::size_t idx) -> LstmTensor
        {
            model.reset_state();
            return LstmTensor(
                encode_sample(Tensor(s), encoding, seed + static_cast<std::uint32_t>(idx)));
        });

    const auto t0 = std::chrono::steady_clock::now();
    const auto epoch_results = trainer.fit_autoencoder(train_backend_samples, val_backend_samples);
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // Inference timing
    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        model.reset_state();
        (void) model.forward(LstmTensor(encoded), false);
    }
    const auto infer_end = std::chrono::steady_clock::now();
    infer_ms = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();

    RunMetrics metrics = evaluate_lstm(model,
        val_samples,
        std::vector<int>(val_samples.size(), 0),
        cfg.training.max_reconstruct_mean_deviation,
        estimate_lstm_macs(make_lstm_cfg(cfg)),
        parameter_count(model.params()),
        encoding,
        seed,
        infer_ms);

    EpochHistory history;
    for (const auto& er : epoch_results)
    {
        history.epoch_nums.push_back(static_cast<float>(er.epoch));
        history.train_losses.push_back(er.train_loss);
        history.val_losses.push_back(er.val_loss);
    }

    const int batches_per_epoch = train_samples.size() / std::max(1, cfg.training.samples_per_batch);
    int batch_idx = 0;
    for (const auto& batch_loss : batch_collector->batch_losses)
    {
        const int current_epoch = (batch_idx / std::max(1, batches_per_epoch)) + 1;
        history.batch_losses.push_back(batch_loss);
        history.batch_epochs.push_back(static_cast<float>(current_epoch));
        ++batch_idx;
    }

    return TrainResult{metrics, history};
}

// ---------------------------------------------------------------------------
// SNN training
// ---------------------------------------------------------------------------

auto train_with_early_stopping_snn(ProtocolSpikingAutoencoder& model,
    Adam& /*optimizer*/,
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
    float& infer_ms) -> TrainResult
{
    // SNN biophysical params (R, C, V_th) need ~10× smaller lr than weights [37].
    nn::training::TrainerConfig tcfg = make_trainer_config(cfg, 0.1F);

    nn::training::Trainer<ProtocolSpikingAutoencoder> trainer(model, tcfg);

    const std::string label =
        "SNN [" + encoding + "] r" + std::to_string(run_id + 1) + "/" + std::to_string(total_runs);
    trainer.add_callback(std::make_shared<nn::training::ProgressCallback>(label));

    auto stopper =
        std::make_shared<nn::training::EarlyStoppingCallback>(cfg.training.early_stop_patience);
    trainer.add_callback(stopper);

    auto batch_collector = std::make_shared<BatchLossCollector>();
    trainer.add_callback(batch_collector);

    std::vector<SnnTensor> train_backend_samples;
    train_backend_samples.reserve(train_samples.size());
    for (const auto& sample : train_samples) train_backend_samples.emplace_back(sample);

    std::vector<SnnTensor> val_backend_samples;
    val_backend_samples.reserve(val_samples.size());
    for (const auto& sample : val_samples) val_backend_samples.emplace_back(sample);

    trainer.set_sample_transform(
        [&encoding, &architecture, alpha, v_th, seed](
            const SnnTensor& s, std::size_t idx) -> SnnTensor
        {
            Tensor enc = encode_sample(Tensor(s), encoding, seed + static_cast<std::uint32_t>(idx));
            enc = apply_snn_architecture_transform(enc, architecture, alpha, v_th);
            return SnnTensor(flatten_time_series(enc));
        });

    const auto t0 = std::chrono::steady_clock::now();
    const auto epoch_results = trainer.fit_autoencoder(train_backend_samples, val_backend_samples);
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // Inference timing
    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        nn::Tensor enc =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        enc = apply_snn_architecture_transform(enc, architecture, alpha, v_th);
        const nn::Tensor flat = flatten_time_series(enc);
        model.reset_state();
        (void) model.forward(SnnTensor(flat), false);
    }
    const auto infer_end = std::chrono::steady_clock::now();
    infer_ms = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();

    const auto enc_sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    const int top_hidden = enc_sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                                   cfg.model.decoder_layer_spec)
                                             : enc_sizes.front();

    RunMetrics metrics = evaluate_snn(model,
        val_samples,
        val_labels,
        cfg.training.max_reconstruct_mean_deviation,
        estimate_snn_macs(static_cast<std::size_t>(cfg.dataset.window_size),
            top_hidden,
            static_cast<int>(enc_sizes.size())),
        parameter_count(model.params()),
        encoding,
        architecture,
        alpha,
        v_th,
        seed,
        infer_ms);

    EpochHistory history;
    for (const auto& er : epoch_results)
    {
        history.epoch_nums.push_back(static_cast<float>(er.epoch));
        history.train_losses.push_back(er.train_loss);
        history.val_losses.push_back(er.val_loss);
    }

    const int batches_per_epoch = train_samples.size() / std::max(1, cfg.training.samples_per_batch);
    int batch_idx = 0;
    for (const auto& batch_loss : batch_collector->batch_losses)
    {
        const int current_epoch = (batch_idx / std::max(1, batches_per_epoch)) + 1;
        history.batch_losses.push_back(batch_loss);
        history.batch_epochs.push_back(static_cast<float>(current_epoch));
        ++batch_idx;
    }

    return TrainResult{metrics, history};
}

} // namespace comparative_autoencoder_experiment
