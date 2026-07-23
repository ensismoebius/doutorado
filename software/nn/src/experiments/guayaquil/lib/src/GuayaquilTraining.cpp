#include "../include/GuayaquilTraining.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "../include/GuayaquilBatchLossCollector.hpp"
#include "../include/GuayaquilEncoding.hpp"
#include "../include/GuayaquilEvaluation.hpp"
#include "../include/GuayaquilMetrics.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "training/EarlyStoppingCallback.hpp"
#include "training/ProgressCallback.hpp"

namespace guayaquil
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

auto make_lstm_cfg(const GuayaquilConfig& cfg) -> nn::models::lstm::LSTMAutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    const int derived_hidden = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                                   cfg.model.decoder_layer_spec)
                                             : sizes.front();
    const int derived_latent =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);

    nn::models::lstm::LSTMAutoencoderConfig arch;
    // The window is consumed lstm_frame_size samples per timestep, so the
    // sequence is that many times shorter. See to_lstm_frames().
    arch.input_size = cfg.model.lstm_frame_size;
    arch.seq_len = cfg.dataset.window_size / cfg.model.lstm_frame_size;
    arch.hidden_size =
        (cfg.model.lstm_hidden_size > 0) ? cfg.model.lstm_hidden_size : derived_hidden;
    arch.latent_size = (cfg.model.latent_dim > 0) ? cfg.model.latent_dim : derived_latent;
    arch.num_layers = static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    return arch;
}

auto make_snn_cfg(const GuayaquilConfig& cfg, float alpha, float v_th) -> AutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    const int effective_l = static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    const int derived_hidden = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                                   cfg.model.decoder_layer_spec)
                                             : sizes.front();
    const int derived_latent =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);

    AutoencoderConfig model_cfg;
    if (cfg.model.loss_type.empty())
        throw std::invalid_argument(
            "GuayaquilTraining: model.loss_function is empty — refusing to guess a "
            "reconstruction loss. Set it explicitly in the profile.");
    model_cfg.loss_type = cfg.model.loss_type;
    // After flatten_time_series, input is {1, window_size*1} — SNN sees window_size features.
    model_cfg.input_features = cfg.dataset.window_size;
    model_cfg.hidden_size =
        (cfg.model.lstm_hidden_size > 0) ? cfg.model.lstm_hidden_size : derived_hidden;
    model_cfg.latent_size = (cfg.model.latent_dim > 0) ? cfg.model.latent_dim : derived_latent;
    model_cfg.depth = effective_l;
    model_cfg.layer_sizes = sizes;
    model_cfg.branch_hidden_size = cfg.model.branch_hidden_size;
    model_cfg.fusion_hidden_size = cfg.model.fusion_hidden_size;
    // Guayaquil's SNN input is flattened by flatten_time_series into a single
    // {1, window_size} frame, so this stack genuinely has ONE time step. Declared
    // explicitly: LifBPTT unrolls exactly one step here, matching the single-step Lif
    // this experiment used before. Left unset it would raise, which is the point.
    model_cfg.time_steps = 1;
    model_cfg.delta_t = 1.0f;
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
// Build a TrainerConfig from GuayaquilConfig
// ---------------------------------------------------------------------------

static auto make_trainer_config(const GuayaquilConfig& cfg, float snn_lr_scale = 1.0F)
    -> nn::training::TrainerConfig
{
    nn::training::TrainerConfig tcfg;
    tcfg.epochs = cfg.training.epochs;
    tcfg.batch_size = std::max(1, cfg.training.samples_per_batch);
    tcfg.learning_rate = cfg.training.learning_rate;
    tcfg.adam_beta1 = cfg.training.beta1;
    tcfg.adam_beta2 = cfg.training.beta2;
    tcfg.adam_epsilon = cfg.training.epsilon;
    // If explicit biophysical lr set in profile, derive scale from it.
    tcfg.snn_lr_scale = (cfg.training.learning_rate_biophysical > 0.0f)
                            ? cfg.training.learning_rate_biophysical / cfg.training.learning_rate
                            : snn_lr_scale;
    return tcfg;
}

// ---------------------------------------------------------------------------
// LSTM training
// ---------------------------------------------------------------------------

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const GuayaquilConfig& cfg,
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

    const std::string label = "LSTM-AE: encoding=" + encoding;
    auto lstm_cb = std::make_shared<nn::training::ProgressCallback>(label);
    lstm_cb->set_metadata(
        "LSTM Autoencoder", static_cast<int>(run_id + 1), static_cast<int>(total_runs), "MSE");
    trainer.add_callback(lstm_cb);

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
    const int lstm_frame = cfg.model.lstm_frame_size;
    trainer.set_sample_transform(
        [&model, &encoding, seed, lstm_frame](const LstmTensor& s, std::size_t idx) -> LstmTensor
        {
            model.reset_state();
            // Encode on the flat window, then frame — the encodings operate on
            // the (window_size, 1) layout.
            return LstmTensor(to_lstm_frames(
                encode_sample(Tensor(s), encoding, seed + static_cast<std::uint32_t>(idx)),
                lstm_frame));
        });

    const auto t0 = std::chrono::steady_clock::now();
    const auto epoch_results = trainer.fit_autoencoder(train_backend_samples, val_backend_samples);
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // Inference timing
    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded = to_lstm_frames(
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i)),
            lstm_frame);
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
        infer_ms,
        lstm_frame);

    EpochHistory history;
    for (const auto& er : epoch_results)
    {
        history.epoch_nums.push_back(static_cast<float>(er.epoch));
        history.train_losses.push_back(er.train_loss);
        history.val_losses.push_back(er.val_loss);
    }

    const int batches_per_epoch =
        train_samples.size() / std::max(1, cfg.training.samples_per_batch);
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
    const GuayaquilConfig& cfg,
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

    std::ostringstream lbl;
    lbl << "SNN-" << architecture << ": encoding=" << encoding << "  v=" << std::fixed
        << std::setprecision(1) << v_th << "  a=" << std::setprecision(2) << alpha;
    auto snn_cb = std::make_shared<nn::training::ProgressCallback>(lbl.str());
    snn_cb->set_metadata("SNN Autoencoder (" + architecture + ")",
        static_cast<int>(run_id + 1),
        static_cast<int>(total_runs),
        "MSE");
    trainer.add_callback(snn_cb);

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

    const int batches_per_epoch =
        train_samples.size() / std::max(1, cfg.training.samples_per_batch);
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

} // namespace guayaquil
