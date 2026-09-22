#include "../include/Meeting01Training.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "../include/Meeting01BatchLossCollector.hpp"
#include "../include/Meeting01Encoding.hpp"
#include "../include/Meeting01Evaluation.hpp"
#include "../include/Meeting01Metrics.hpp"
#include "Meeting01AeCommon.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "layers/spiking/ArcTanSurrogate.hpp"
#include "training/EarlyStoppingCallback.hpp"
#include "training/ProgressCallback.hpp"

using nn::models::autoencoder::AutoencoderConfig;
using nn::models::autoencoder::ProtocolSpikingAutoencoder;

namespace meeting01
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

auto make_lstm_cfg(const Meeting01Config& cfg) -> nn::models::lstm::LSTMAutoencoderConfig
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

namespace
{
// Renders free-form encoder_widths into the same "linear:W:act" spec convention the
// profile's fixed encoder_layer_spec/decoder_layer_spec use — mirrors
// pga::to_ae_config's rendering (GaGenome.cpp) so a genome-driven and a
// profile-driven stack are built by the exact same downstream parser.
auto render_encoder_spec(const std::vector<int>& widths) -> std::vector<std::string>
{
    std::vector<std::string> spec;
    for (std::size_t i = 0; i + 1 < widths.size(); ++i)
        spec.push_back("linear:" + std::to_string(widths[i]) + ":leaky");
    spec.push_back("linear:" + std::to_string(widths.back()) + ":identity");
    return spec;
}

auto render_decoder_spec(const std::vector<int>& widths) -> std::vector<std::string>
{
    std::vector<std::string> spec;
    for (std::size_t i = widths.size() - 1; i-- > 0;)
        spec.push_back("linear:" + std::to_string(widths[i]) + ":leaky");
    spec.push_back("linear:output:identity");
    return spec;
}
} // namespace

auto make_snn_cfg(
    const Meeting01Config& cfg, float alpha, float v_th, const std::vector<int>& encoder_widths)
    -> AutoencoderConfig
{
    const auto sizes =
        encoder_widths.empty() ? extract_layer_sizes(cfg.model.encoder_layer_spec) : encoder_widths;
    const int effective_l = static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    const int derived_hidden = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                                   cfg.model.decoder_layer_spec)
                                             : sizes.front();
    const int derived_latent =
        encoder_widths.empty()
            ? extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec)
            : encoder_widths.back();

    AutoencoderConfig model_cfg;
    if (cfg.model.loss_type.empty())
        throw std::invalid_argument(
            "Meeting01Training: model.loss_function is empty — refusing to guess a "
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
    // The encoder emits a time-major (T*B, window_size) tensor, so LifBPTT unrolls T
    // real steps and the membrane actually carries state between them. This was 1 until
    // 2026-09-22; at T=1 `beta` multiplied a zero-initialised membrane, which made both
    // `alpha` and `v_th` inert, and rate/latency coding have no time axis to live on.
    model_cfg.time_steps = cfg.model.time_steps;
    model_cfg.delta_t = 1.0f;
    // R and C are chosen so the two knobs stay independent and mean what they say:
    // with delta_t = 1 and R = 1, beta = exp(-1/C) = alpha exactly, while v_th is the
    // firing threshold instead of being folded into the time constant via R = 1/v_th.
    model_cfg.resistance = 1.0f;
    model_cfg.capacitance = std::max(1e-3f, -1.0f / std::log(std::clamp(alpha, 1e-3f, 0.999f)));
    model_cfg.voltage_threshold = std::max(v_th, 1e-3f);

    if (!encoder_widths.empty())
    {
        model_cfg.encoder_layer_spec = render_encoder_spec(encoder_widths);
        model_cfg.decoder_layer_spec = render_decoder_spec(encoder_widths);
    }
    else
    {
        model_cfg.encoder_layer_spec =
            cfg.model.encoder_layer_spec.empty()
                ? std::vector<std::string>{"linear:hidden:leaky", "linear:latent:identity"}
                : cfg.model.encoder_layer_spec;

        model_cfg.decoder_layer_spec =
            cfg.model.decoder_layer_spec.empty()
                ? std::vector<std::string>{"linear:hidden:leaky", "linear:output:identity"}
                : cfg.model.decoder_layer_spec;
    }

    model_cfg.branch_encoder_layer_spec = cfg.model.branch_encoder_layer_spec;
    model_cfg.branch_decoder_layer_spec = cfg.model.branch_decoder_layer_spec;
    model_cfg.fusion_encoder_layer_spec = cfg.model.fusion_encoder_layer_spec;
    model_cfg.fusion_decoder_layer_spec = cfg.model.fusion_decoder_layer_spec;
    // ArcTan over the framework's ExponentialSurrogate default: snnTorch's default
    // spike-derivative estimator since 2023, with heavier gradient tails that avoid
    // the saturation exponential/boxcar surrogates show away from threshold.
    model_cfg.surrogate_gradient = std::make_shared<ArcTanSurrogate>();
    return model_cfg;
}

// ---------------------------------------------------------------------------
// Build a TrainerConfig from Meeting01Config
// ---------------------------------------------------------------------------

static auto make_trainer_config(const Meeting01Config& cfg, float snn_lr_scale = 1.0F)
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
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult
{
    return train_ae(model,
        cfg,
        train_samples,
        val_samples,
        encoding,
        seed,
        run_id,
        total_runs,
        "LSTM-AE: encoding=" + encoding,
        "LSTM Autoencoder",
        estimate_lstm_macs(make_lstm_cfg(cfg)),
        train_ms,
        infer_ms);
}

// ---------------------------------------------------------------------------
// GRU / Transformer training — same frame-consuming path as the LSTM-AE.
// ---------------------------------------------------------------------------

auto make_gru_cfg(const Meeting01Config& cfg) -> nn::models::gru::GRUAutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    const int derived_hidden = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                                   cfg.model.decoder_layer_spec)
                                             : sizes.front();
    const int derived_latent =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);

    nn::models::gru::GRUAutoencoderConfig arch;
    arch.input_size = cfg.model.lstm_frame_size;
    arch.seq_len = cfg.dataset.window_size / cfg.model.lstm_frame_size;
    arch.hidden_size =
        (cfg.model.lstm_hidden_size > 0) ? cfg.model.lstm_hidden_size : derived_hidden;
    arch.latent_size = (cfg.model.latent_dim > 0) ? cfg.model.latent_dim : derived_latent;
    arch.num_layers = static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    return arch;
}

auto make_transformer_cfg(const Meeting01Config& cfg)
    -> nn::models::transformer::TransformerAutoencoderConfig
{
    const int derived_latent =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);

    nn::models::transformer::TransformerAutoencoderConfig arch;
    arch.input_size = cfg.model.lstm_frame_size;
    arch.seq_len = cfg.dataset.window_size / cfg.model.lstm_frame_size;
    arch.d_model = cfg.model.transformer_d_model;
    arch.n_heads = cfg.model.transformer_heads;
    arch.n_layers = cfg.model.transformer_layers;
    arch.d_ff = cfg.model.transformer_d_ff;
    arch.latent_size = (cfg.model.latent_dim > 0) ? cfg.model.latent_dim : derived_latent;
    return arch;
}

auto train_with_early_stopping_gru(nn::models::gru::GRUAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult
{
    return train_ae(model,
        cfg,
        train_samples,
        val_samples,
        encoding,
        seed,
        run_id,
        total_runs,
        "GRU-AE: encoding=" + encoding,
        "GRU Autoencoder",
        estimate_gru_macs(make_gru_cfg(cfg)),
        train_ms,
        infer_ms);
}

auto train_with_early_stopping_transformer(nn::models::transformer::TransformerAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult
{
    return train_ae(model,
        cfg,
        train_samples,
        val_samples,
        encoding,
        seed,
        run_id,
        total_runs,
        "Transformer-AE: encoding=" + encoding,
        "Transformer Autoencoder",
        estimate_transformer_macs(make_transformer_cfg(cfg)),
        train_ms,
        infer_ms);
}

// ---------------------------------------------------------------------------
// SNN training
// ---------------------------------------------------------------------------

auto train_with_early_stopping_snn(ProtocolSpikingAutoencoder& model,
    const Meeting01Config& cfg,
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

    // Plain per-epoch log lines (survive nohup / pipes, where the live bars collapse).
    trainer.add_callback(std::make_shared<Meeting01EpochLogger>(
        progress_context(cfg, run_id, total_runs, seed), lbl.str()));

    // Structured JSONL events for the live monitor (identity from the driver's
    // pending context). No-op when the events sink was never opened.
    trainer.add_callback(std::make_shared<Meeting01EventCallback>());

    auto stopper =
        std::make_shared<nn::training::EarlyStoppingCallback>(cfg.training.early_stop_patience);
    trainer.add_callback(stopper);

    auto batch_collector = std::make_shared<BatchLossCollector>();
    trainer.add_callback(batch_collector);

    const int steps = cfg.model.time_steps;

    // Input is the encoded (time_steps, window_size) spike tensor; target is the
    // ORIGINAL analog window held across those same steps. Reconstructing the encoded
    // input instead made the loss incomparable across encodings and handed the GA a
    // free win for picking latency (see .wiki/Experiments/Meeting01.md).
    using SnnPair = std::pair<SnnTensor, SnnTensor>;
    auto make_pairs = [&](const std::vector<Tensor>& src)
    {
        std::vector<SnnPair> pairs;
        pairs.reserve(src.size());
        for (std::size_t i = 0; i < src.size(); ++i)
        {
            Tensor enc =
                encode_sample(src[i], encoding, seed + static_cast<std::uint32_t>(i), steps);
            enc = apply_snn_architecture_transform(enc, architecture, alpha, v_th);
            pairs.emplace_back(
                SnnTensor(enc), SnnTensor(make_reconstruction_target(src[i], steps)));
        }
        return pairs;
    };
    const auto train_pairs = make_pairs(train_samples);
    const auto val_pairs = make_pairs(val_samples);

    const auto t0 = std::chrono::steady_clock::now();
    const auto epoch_results = trainer.fit_supervised(train_pairs, val_pairs);
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // Inference timing
    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        nn::Tensor enc =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i), steps);
        enc = apply_snn_architecture_transform(enc, architecture, alpha, v_th);
        model.reset_state();
        (void) model.forward(SnnTensor(enc), false);
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
        infer_ms,
        steps);

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

} // namespace meeting01
