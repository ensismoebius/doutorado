#include "../include/Meeting01Training.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "../include/Meeting01BatchLossCollector.hpp"
#include "../include/Meeting01Encoding.hpp"
#include "../include/Meeting01Energy.hpp"
#include "../include/Meeting01Evaluation.hpp"
#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01Stopping.hpp"
#include "Meeting01AeCommon.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "layers/spiking/ArcTanSurrogate.hpp"
#include "training/BestWeightsCallback.hpp"
#include "training/EarlyStoppingCallback.hpp"
#include "training/ProgressCallback.hpp"
#include "utility/GaussianNoise.hpp"

using nn::models::autoencoder::AutoencoderConfig;
using nn::models::autoencoder::ProtocolSpikingAutoencoder;

namespace meeting01
{

using LstmTensor = nn::models::lstm::LSTMAutoencoder::Tensor;
using SnnTensor = ProtocolSpikingAutoencoder::Tensor;

// ---------------------------------------------------------------------------
// Config helpers (unchanged)
// ---------------------------------------------------------------------------

namespace
{
// The width field of a layer spec ("linear:64:leaky" -> 64). A field that is not a whole
// positive integer is an error naming the spec (it used to be skipped, silently dropping the
// layer from the derived sizes).
auto parse_spec_width(const std::string& spec, const std::string& field) -> int
{
    std::size_t used = 0;
    int width = 0;
    try
    {
        width = std::stoi(field, &used);
    }
    catch (const std::exception&)
    {
        used = 0;
    }
    if (used == 0 || used != field.size() || width <= 0)
    {
        throw std::invalid_argument(
            "layer spec '" + spec + "': width '" + field + "' is not a positive integer");
    }
    return width;
}
} // namespace

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
            sizes.push_back(parse_spec_width(spec, parts[1]));
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
        // A bare token (`time_mean`, `residual`) has no width: -1 sends the search on.
        return (parts.size() >= 2) ? parse_spec_width(spec, parts[1]) : -1;
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
    // No width in either spec: refuse instead of inventing one (this returned 16).
    throw std::invalid_argument(
        "cannot derive the latent size: neither the last encoder spec nor the first decoder "
        "spec declares a width (e.g. linear:16:identity); set model.latent_dim");
}

auto make_lstm_cfg(const Meeting01Config& cfg, int hidden_size_override, int num_layers_override)
    -> nn::models::lstm::LSTMAutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    const int derived_hidden = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                                   cfg.model.decoder_layer_spec)
                                             : sizes.front();
    const int derived_latent =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);

    nn::models::lstm::LSTMAutoencoderConfig arch;
    // encode_sample() (2026-09-22, commit 19c7b488) expands every window into a
    // (time_steps, window_size) time-major tensor BEFORE to_lstm_frames() reframes it —
    // this is the same real temporal axis the SNN unrolls, not a baseline-only
    // shortcut, so the actual sequence length to_lstm_frames() produces is
    // (window_size * time_steps) / lstm_frame_size, not window_size / lstm_frame_size.
    // Getting this wrong doesn't affect LSTMAutoencoder's own forward pass (it infers
    // seq_len dynamically from the encoder's actual input via last_T_), but it DOES
    // silently under-count estimate_lstm_macs()'s cost (linear in seq_len) — the same
    // stale formula in TransformerAutoencoderConfig crashes outright, since that model's
    // positional encoding buffer is sized once from this field at construction.
    arch.input_size = cfg.model.lstm_frame_size;
    arch.seq_len =
        (cfg.dataset.window_size * cfg.model.recurrent_time_steps) / cfg.model.lstm_frame_size;
    arch.hidden_size = (hidden_size_override > 0)         ? hidden_size_override
                       : (cfg.model.lstm_hidden_size > 0) ? cfg.model.lstm_hidden_size
                                                          : derived_hidden;
    // latent_dim is never genome-driven — fixed compression ratio across every family.
    arch.latent_size = (cfg.model.latent_dim > 0) ? cfg.model.latent_dim : derived_latent;
    arch.num_layers = (num_layers_override > 0)
                          ? num_layers_override
                          : static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    return arch;
}

namespace
{
// Renders free-form encoder_widths into the same "linear:W:act" spec convention the
// profile's fixed encoder_layer_spec/decoder_layer_spec use — mirrors
// pga::to_ae_config's rendering (GaGenome.cpp) so a genome-driven and a
// profile-driven stack are built by the exact same downstream parser.
//
// `hidden` is the hidden layers' activation token ("leaky" is a LIF in the SNN builder and a
// leaky ReLU in the ANN one; "leaky_relu" is a leaky ReLU in both); `time_mean` appends the
// TimeMeanPool stage (SNN builder only).
auto render_encoder_spec(const std::vector<int>& widths, bool time_mean) -> std::vector<std::string>
{
    std::vector<std::string> spec;
    for (std::size_t i = 0; i + 1 < widths.size(); ++i)
        spec.push_back("linear:" + std::to_string(widths[i]) + ":leaky");
    spec.push_back("linear:" + std::to_string(widths.back()) + ":identity");
    if (time_mean) spec.emplace_back("time_mean");
    return spec;
}

auto render_decoder_spec(const std::vector<int>& widths, const std::string& hidden, bool time_mean)
    -> std::vector<std::string>
{
    std::vector<std::string> spec;
    for (std::size_t i = widths.size() - 1; i-- > 0;)
        spec.push_back("linear:" + std::to_string(widths[i]) + ":" + hidden);
    spec.push_back("linear:output:identity");
    if (time_mean) spec.emplace_back("time_mean");
    return spec;
}
} // namespace

auto snn_output_steps(const Meeting01Config& cfg) -> int
{
    return cfg.model.snn_time_pool == "none" ? cfg.model.time_steps : 1;
}

auto snn_time_steps_range(const Meeting01Config& cfg) -> std::pair<int, int>
{
    const auto& ga = cfg.evaluation.ga.snn;
    if (ga.time_steps_min == 0 && ga.time_steps_max == 0)
        return {cfg.model.time_steps, cfg.model.time_steps};
    return {ga.time_steps_min, ga.time_steps_max};
}

auto with_snn_time_steps(const Meeting01Config& cfg, int time_steps) -> Meeting01Config
{
    const auto [lo, hi] = snn_time_steps_range(cfg);
    if (time_steps < lo || time_steps > hi)
        throw std::invalid_argument("with_snn_time_steps: T = " + std::to_string(time_steps) +
                                    " is outside the profile's range [" + std::to_string(lo) +
                                    ", " + std::to_string(hi) +
                                    "]: a genome or checkpoint from another profile.");
    Meeting01Config out = cfg;
    out.model.time_steps = time_steps;
    return out;
}

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
    // The genes are used as given, never clamped (a clamped alpha would make beta differ
    // from the gene the GA reports): check_ga keeps the profile's ranges inside these
    // limits, so a value outside them is a caller's bug.
    if (!(alpha >= 1e-3f && alpha <= 0.999f))
        throw std::invalid_argument("make_snn_cfg: alpha = " + std::to_string(alpha) +
                                    " is outside [0.001, 0.999], where beta = alpha holds. "
                                    "Keep evaluation.ga.snn.alpha_min/max inside it.");
    if (!(v_th >= 1e-3f))
        throw std::invalid_argument("make_snn_cfg: v_th = " + std::to_string(v_th) +
                                    " is below 0.001. Keep "
                                    "evaluation.ga.snn.voltage_threshold_min >= 0.001.");
    model_cfg.resistance = 1.0f;
    model_cfg.capacitance = -1.0f / std::log(alpha);
    model_cfg.voltage_threshold = v_th;

    if (!encoder_widths.empty())
    {
        const std::string& pool = cfg.model.snn_time_pool;
        model_cfg.encoder_layer_spec = render_encoder_spec(encoder_widths, pool == "latent");
        model_cfg.decoder_layer_spec = render_decoder_spec(encoder_widths,
            cfg.model.snn_decoder_neuron == "analog" ? "leaky_relu" : "leaky",
            pool == "output");
    }
    else
    {
        if (cfg.model.snn_time_pool != "none" || cfg.model.snn_decoder_neuron != "lif")
            throw std::invalid_argument(
                "make_snn_cfg: model.snn_time_pool \"" + cfg.model.snn_time_pool +
                "\" / model.snn_decoder_neuron \"" + cfg.model.snn_decoder_neuron +
                "\" need encoder widths to render the stack from (the GA genome's); the "
                "fixed encoder_layer_spec/decoder_layer_spec path builds the profile specs "
                "verbatim. Pass the widths, or set \"none\" / \"lif\".");
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
    // Empty keeps the builder's own init; ann-ae (make_ann_twin_cfg) inherits this value, so
    // a set model.initializer gives both arms the same one.
    model_cfg.initializer = cfg.model.initializer;
    return model_cfg;
}

auto make_ann_twin_cfg(const Meeting01Config& cfg, const std::vector<int>& encoder_widths)
    -> AutoencoderConfig
{
    if (encoder_widths.empty())
        throw std::invalid_argument(
            "make_ann_twin_cfg: no encoder widths. The ann-ae control copies the SNN winner's "
            "widths, so it needs that winner's genome.");
    // alpha / v_th are spiking-only fields of the shared builder config; the ANN builders ignore
    // them (AutoencoderConfig: "SNN-specific parameters (ignored by ANN models)").
    AutoencoderConfig model_cfg = make_snn_cfg(cfg, 0.9F, 1.0F, encoder_widths);
    model_cfg.encoder_layer_spec = render_encoder_spec(encoder_widths, /*time_mean=*/false);
    model_cfg.decoder_layer_spec =
        render_decoder_spec(encoder_widths, "leaky", /*time_mean=*/false);
    return model_cfg;
}

// ---------------------------------------------------------------------------
// Build a TrainerConfig from Meeting01Config
// ---------------------------------------------------------------------------

// `seed` (required) orders the windows: TrainerConfig::sampler_shuffle_seed (results_format 9;
// before, the default 42 gave every seed the same order).
static auto make_trainer_config(const Meeting01Config& cfg, std::uint32_t seed, float snn_lr_scale)
    -> nn::training::TrainerConfig
{
    nn::training::TrainerConfig tcfg;
    tcfg.epochs = cfg.training.epochs;
    tcfg.batch_size = std::max(1, cfg.training.samples_per_batch);
    tcfg.learning_rate = cfg.training.learning_rate;
    tcfg.adam_beta1 = cfg.training.beta1;
    tcfg.adam_beta2 = cfg.training.beta2;
    tcfg.adam_epsilon = cfg.training.epsilon;
    tcfg.lr_warmup_steps = cfg.training.lr_warmup_steps;
    tcfg.sampler_shuffle_seed = seed;
    // If explicit biophysical lr set in profile, derive scale from it.
    tcfg.snn_lr_scale = (cfg.training.learning_rate_biophysical > 0.0f)
                            ? cfg.training.learning_rate_biophysical / cfg.training.learning_rate
                            : snn_lr_scale;
    return tcfg;
}

// ---------------------------------------------------------------------------
// LSTM training
auto frame_ae_cost(const Meeting01Config& cfg,
    const std::string& encoding,
    std::size_t macs,
    EnergyLedger model_energy) -> ModelCost
{
    // A frame-consuming model reads the whole encoded tensor, so the encoder writes all of it.
    model_energy += encoding_energy(encoding,
        static_cast<double>(cfg.dataset.window_size),
        cfg.model.recurrent_time_steps,
        0.0,
        /*event_driven=*/false);
    return ModelCost{macs, model_energy};
}

// ---------------------------------------------------------------------------

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<WindowMetadata>& train_meta,
    const std::vector<WindowMetadata>& val_meta,
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
        train_meta,
        val_meta,
        encoding,
        seed,
        run_id,
        total_runs,
        "LSTM-AE: encoding=" + encoding,
        "LSTM Autoencoder",
        frame_ae_cost(cfg,
            encoding,
            estimate_lstm_macs(make_lstm_cfg(cfg)),
            lstm_ae_energy(make_lstm_cfg(cfg))),
        train_ms,
        infer_ms);
}

// ---------------------------------------------------------------------------
// GRU / Transformer training — same frame-consuming path as the LSTM-AE.
// ---------------------------------------------------------------------------

auto make_gru_cfg(const Meeting01Config& cfg, int hidden_size_override, int num_layers_override)
    -> nn::models::gru::GRUAutoencoderConfig
{
    const auto sizes = extract_layer_sizes(cfg.model.encoder_layer_spec);
    const int derived_hidden = sizes.empty() ? extract_latent_size(cfg.model.encoder_layer_spec,
                                                   cfg.model.decoder_layer_spec)
                                             : sizes.front();
    const int derived_latent =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);

    nn::models::gru::GRUAutoencoderConfig arch;
    arch.input_size = cfg.model.lstm_frame_size;
    // See make_lstm_cfg()'s comment: to_lstm_frames() reframes the (time_steps,
    // window_size) tensor encode_sample() already produced, so the real sequence
    // length includes the time_steps factor too.
    arch.seq_len =
        (cfg.dataset.window_size * cfg.model.recurrent_time_steps) / cfg.model.lstm_frame_size;
    arch.hidden_size = (hidden_size_override > 0)         ? hidden_size_override
                       : (cfg.model.lstm_hidden_size > 0) ? cfg.model.lstm_hidden_size
                                                          : derived_hidden;
    arch.latent_size = (cfg.model.latent_dim > 0) ? cfg.model.latent_dim : derived_latent;
    arch.num_layers = (num_layers_override > 0)
                          ? num_layers_override
                          : static_cast<int>(std::max<std::size_t>(1, sizes.size()));
    return arch;
}

auto make_transformer_cfg(const Meeting01Config& cfg,
    int d_model_override,
    int n_heads_override,
    int n_layers_override,
    int d_ff_override) -> nn::models::transformer::TransformerAutoencoderConfig
{
    const int derived_latent =
        extract_latent_size(cfg.model.encoder_layer_spec, cfg.model.decoder_layer_spec);

    nn::models::transformer::TransformerAutoencoderConfig arch;
    arch.input_size = cfg.model.lstm_frame_size;
    // See make_lstm_cfg()'s comment. Unlike LSTM/GRU, TransformerAutoencoder sizes its
    // positional-encoding buffer from this field ONCE at construction (it does not
    // infer seq_len dynamically) — getting it wrong here doesn't just under-cost
    // estimate_transformer_macs(), it makes every real forward pass overrun that buffer
    // and throw "Block indices out of range".
    arch.seq_len =
        (cfg.dataset.window_size * cfg.model.recurrent_time_steps) / cfg.model.lstm_frame_size;
    arch.d_model = (d_model_override > 0) ? d_model_override : cfg.model.transformer_d_model;
    arch.n_heads = (n_heads_override > 0) ? n_heads_override : cfg.model.transformer_heads;
    arch.n_layers = (n_layers_override > 0) ? n_layers_override : cfg.model.transformer_layers;
    arch.d_ff = (d_ff_override > 0) ? d_ff_override : cfg.model.transformer_d_ff;
    arch.norm_first = cfg.model.transformer_norm_first;
    arch.latent_size = (cfg.model.latent_dim > 0) ? cfg.model.latent_dim : derived_latent;
    return arch;
}

auto train_with_early_stopping_gru(nn::models::gru::GRUAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<WindowMetadata>& train_meta,
    const std::vector<WindowMetadata>& val_meta,
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
        train_meta,
        val_meta,
        encoding,
        seed,
        run_id,
        total_runs,
        "GRU-AE: encoding=" + encoding,
        "GRU Autoencoder",
        frame_ae_cost(
            cfg, encoding, estimate_gru_macs(make_gru_cfg(cfg)), gru_ae_energy(make_gru_cfg(cfg))),
        train_ms,
        infer_ms);
}

auto train_with_early_stopping_transformer(nn::models::transformer::TransformerAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<WindowMetadata>& train_meta,
    const std::vector<WindowMetadata>& val_meta,
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
        train_meta,
        val_meta,
        encoding,
        seed,
        run_id,
        total_runs,
        "Transformer-AE: encoding=" + encoding,
        "Transformer Autoencoder",
        frame_ae_cost(cfg,
            encoding,
            estimate_transformer_macs(make_transformer_cfg(cfg)),
            transformer_ae_energy(make_transformer_cfg(cfg))),
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
    const std::vector<WindowMetadata>& train_meta,
    const std::vector<WindowMetadata>& val_meta,
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
    nn::training::TrainerConfig tcfg = make_trainer_config(cfg, seed, 0.1F);

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
    auto events = std::make_shared<Meeting01EventCallback>();
    trainer.add_callback(events);

    // The stopping callbacks need the monitor set's mean-predictor loss, so they are added
    // once the triples below exist.

    auto batch_collector = std::make_shared<BatchLossCollector>();
    trainer.add_callback(batch_collector);

    const int steps = cfg.model.time_steps;
    const int out_steps = snn_output_steps(cfg);

    // Denoising-autoencoder corruption (Vincent et al. 2008/2010 — see
    // .wiki/Core/DataLoaders.md#denoising-autoencoder-corruption): additive Gaussian noise
    // on the encoder's input only, fixed for this run/seed (not resampled per epoch — see
    // that wiki section for why, mirroring the poisson-encoding precedent below).
    // denoising_noise_std == 0 (default) makes this an exact no-op, byte-identical to
    // before this feature existed.
    const nn::transforms::GaussianNoise denoise_noise(cfg.model.denoising_noise_std, seed);

    // Input is the encoded (time_steps, window_size) spike tensor; target is the
    // ORIGINAL analog window, held across those same steps when the model emits one row
    // per step (snn_time_pool "none") and given once when it averages them (out_steps 1).
    // Reconstructing the encoded input instead made the loss incomparable across encodings
    // and handed the GA a free win for picking latency (see .wiki/Experiments/Meeting01.md).
    //
    // The mask excludes the zero-padded tail of a variable-length window's last slice
    // (FSDD/AudioMNIST only — see WindowMetadata::valid_length) from the loss.
    using SnnTriple = nn::training::Trainer<ProtocolSpikingAutoencoder>::SampleTriple;
    auto make_triples = [&](const std::vector<Tensor>& src,
                            const std::vector<WindowMetadata>& meta,
                            bool apply_noise)
    {
        std::vector<SnnTriple> triples;
        triples.reserve(src.size());
        for (std::size_t i = 0; i < src.size(); ++i)
        {
            // Corruption feeds the encoder only — target and mask below are always built
            // from the CLEAN src[i], never the noisy version.
            const Tensor encoder_input = apply_noise ? denoise_noise(src[i]) : src[i];
            Tensor enc =
                encode_sample(encoder_input, encoding, seed + static_cast<std::uint32_t>(i), steps);
            enc = apply_snn_architecture_transform(enc, architecture, alpha, v_th);
            const Tensor mask = make_reconstruction_target(
                make_activity_mask(meta[i].valid_length, static_cast<int>(src[i].size())),
                out_steps);
            triples.emplace_back(SnnTensor(enc),
                SnnTensor(make_reconstruction_target(src[i], out_steps)),
                SnnTensor(mask));
        }
        return triples;
    };
    const auto train_triples = make_triples(train_samples, train_meta, /*apply_noise=*/true);
    const auto val_triples = make_triples(val_samples, val_meta, /*apply_noise=*/false);

    // Patience armed by skill over the mean predictor, best epoch kept (Meeting01Stopping.hpp).
    const StoppingRule rule = make_stopping_rule(cfg,
        input_independent_monitor_loss(train_triples, val_triples, trainer.config().batch_size));
    events->set_stopping_rule(rule.trivial_loss, rule.arm_below, cfg.training.early_stop_min_skill);
    trainer.add_callback(std::make_shared<nn::training::EarlyStoppingCallback>(
        cfg.training.early_stop_patience, 1e-8F, rule.arm_below));
    trainer.add_callback(
        std::make_shared<nn::training::BestWeightsCallback<ProtocolSpikingAutoencoder>>(model));

    const auto t0 = std::chrono::steady_clock::now();
    const auto epoch_results = trainer.fit_supervised_masked(train_triples, val_triples);
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

    RunMetrics metrics = evaluate_snn(model,
        val_samples,
        val_meta,
        val_labels,
        cfg.training.max_reconstruct_mean_deviation,
        parameter_count(model.params()),
        encoding,
        architecture,
        alpha,
        v_th,
        seed,
        infer_ms,
        steps,
        out_steps);

    EpochHistory history;
    for (const auto& er : epoch_results)
    {
        history.epoch_nums.push_back(static_cast<float>(er.epoch));
        history.train_losses.push_back(er.train_loss);
        history.val_losses.push_back(er.val_loss);
    }

    const int batches_per_epoch =
        static_cast<int>(train_samples.size()) / std::max(1, cfg.training.samples_per_batch);
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
