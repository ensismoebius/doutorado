// Internal (not public API) helpers behind extract_features()'s autoencoder
// strategy. Split out of ThesisFeatureExtraction.cpp purely to keep that file
// under the project's file-length limit -- everything here is reachable from
// run_protocol_ae<AEType>'s template instantiation chain (directly or via
// build_ae_config/build_spike_frames/encode_mean_latents), so it has to be
// visible wherever that template is instantiated. Kept as a single unnamed
// namespace, exactly as it was inline in the .cpp: this header is included by
// exactly one translation unit (ThesisFeatureExtraction.cpp), so an unnamed
// namespace here behaves identically to writing this code directly in that
// .cpp -- no linkage change, no ODR risk, pure code motion.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ThesisFeatureExtraction.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "layers/losses/MAELoss.hpp"        // MAE reconstruction loss (Trainer.hpp only pulls MSE)
#include "layers/losses/SpikeCountLoss.hpp" // rate-coded (poisson) reconstruction
#include "layers/losses/SpikeTimeLoss.hpp"  // latency-coded reconstruction
#include "models/autoencoder/ProtocolAutoencoder.hpp"        // ANN-AE (non-spiking)
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp" // SNN-AE (spiking)
#include "training/ProgressCallback.hpp"

namespace thesis
{

using nn::models::autoencoder::AutoencoderConfig;

// ─── batch extraction ───────────────────────────────────────────────────────

namespace
{
std::vector<double> tensor_to_vec(const nn::Tensor& t)
{
    std::vector<double> v;
    v.reserve(static_cast<size_t>(t.size()));
    for (long i = 0; i < t.rows(); ++i)
        for (long j = 0; j < t.cols(); ++j) v.push_back(static_cast<double>(t.at(i, j)));
    return v;
}

// Pre-emphasis coefficient for the voice signal (first-order high-boost filter
// y[n] = x[n] - alpha*x[n-1]). Compensates the ~-6 dB/octave glottal-source
// spectral tilt so the speaker-discriminative upper formants are not attenuated
// before feature extraction. Applied to audio only, never to EEG.
constexpr double kPreEmphasisAlpha = 0.97;

// Raw-signal accessors, one per recorded modality. Each returns 256 zeros when
// the requested tensor is absent, so downstream transforms always receive a
// non-empty signal. Pre-emphasis is audio-only (see kPreEmphasisAlpha above).
using SignalGetter = std::function<std::vector<double>(const ThesisSample&)>;

std::vector<double> voice_signal(const ThesisSample& sample)
{
    auto present = [](const nn::Tensor& t) { return t.rows() > 0 && t.cols() > 0; };
    std::vector<double> sig;
    if (present(sample.audio))
    {
        sig = tensor_to_vec(sample.audio);
        apply_preemphasis(sig, kPreEmphasisAlpha);
    }
    if (sig.empty()) sig.assign(256, 0.0);
    return sig;
}

std::vector<double> eeg_signal(const ThesisSample& sample)
{
    auto present = [](const nn::Tensor& t) { return t.rows() > 0 && t.cols() > 0; };
    std::vector<double> sig;
    if (present(sample.eeg)) sig = tensor_to_vec(sample.eeg);
    if (sig.empty()) sig.assign(256, 0.0);
    return sig;
}

// Early fusion: one signal = voice samples followed by EEG samples, extracted
// through a single DTWPT/autoencoder pass (audit C12). The two halves have
// different native sample rates (44100 Hz voice vs 1024 Hz EEG); the caller
// applies the voice rate to the whole concatenation for frequency-scale mapping
// since voice dominates the sample count (176400 vs 24576 per trial). This is
// an approximation, not a resampling — documented here so it isn't mistaken for
// a physically exact rate.
std::vector<double> fused_early_signal(const ThesisSample& sample)
{
    std::vector<double> sig = voice_signal(sample);
    const std::vector<double> eeg = eeg_signal(sample);
    sig.insert(sig.end(), eeg.begin(), eeg.end());
    return sig;
}

// Nominal sample rates by modality (Hz), per the dataset protocol
// (Pressel Coretto et al., SPIE 2017 / 10.1117/12.2255697).
constexpr double kVoiceSampleRate = 44100.0;
constexpr double kEegSampleRate = 1024.0;

bool is_linear_spec(const std::string& s)
{
    return s.rfind("linear:", 0) == 0;
}

int extract_linear_dim(const std::string& spec, int fallback)
{
    if (!is_linear_spec(spec)) return fallback;
    const auto first_colon = spec.find(':');
    const auto second_colon = spec.find(':', first_colon + 1);
    const std::string token = spec.substr(first_colon + 1, second_colon - first_colon - 1);
    if (token == "output" || token == "N_speakers") return fallback;
    try
    {
        return std::stoi(token);
    }
    catch (...)
    {
        return fallback;
    }
}

int first_encoder_dim(const std::vector<std::string>& spec, int fallback)
{
    for (const auto& s : spec)
    {
        const int dim = extract_linear_dim(s, fallback);
        if (dim != fallback) return dim;
    }
    return fallback;
}

int last_encoder_dim(const std::vector<std::string>& spec, int fallback)
{
    for (auto it = spec.rbegin(); it != spec.rend(); ++it)
    {
        const int dim = extract_linear_dim(*it, fallback);
        if (dim != fallback) return dim;
    }
    return fallback;
}

// Frame a 1-D signal into a (T_frames × frame_len) matrix for the sequence AE:
// row t, column w = sig[t*frame_len + w]. sig must have length >= T_frames*frame_len.
// Windowing keeps the LSTM sequence short (T_frames steps of frame_len features)
// instead of one step-per-sample sequence that is too long to unroll.
nn::Tensor vec_to_frame_tensor(const std::vector<double>& sig, int T_frames, int frame_len)
{
    nn::Tensor t(static_cast<nn::Index>(T_frames), static_cast<nn::Index>(frame_len));
    for (int f = 0; f < T_frames; ++f)
        for (int w = 0; w < frame_len; ++w)
            t.at(static_cast<nn::Index>(f), static_cast<nn::Index>(w)) =
                static_cast<float>(sig[static_cast<size_t>(f) * static_cast<size_t>(frame_len) +
                                       static_cast<size_t>(w)]);
    return t;
}

// Cap on the AE sequence length: signals are windowed into at most this many
// frames, so seq_len stays small regardless of raw signal length.
constexpr int kAeMaxFrames = 64;

// Fixed flat input dimension for the Protocol (SNN-AE / ANN-AE) autoencoders,
// which consume a single vector per sample (not a sequence). The raw signal is
// average-pooled into this many contiguous bins.
constexpr int kAeInputFeatures = 256;

// Average-pool a 1-D signal into `out_dim` contiguous bins.
std::vector<double> pool_signal(const std::vector<double>& sig, int out_dim)
{
    std::vector<double> out(static_cast<size_t>(out_dim), 0.0);
    if (sig.empty()) return out;
    for (int b = 0; b < out_dim; ++b)
    {
        const size_t lo = static_cast<size_t>(b) * sig.size() / static_cast<size_t>(out_dim);
        size_t hi = static_cast<size_t>(b + 1) * sig.size() / static_cast<size_t>(out_dim);
        if (hi <= lo) hi = lo + 1;
        double s = 0.0;
        size_t n = 0;
        for (size_t i = lo; i < hi && i < sig.size(); ++i)
        {
            s += sig[i];
            ++n;
        }
        out[static_cast<size_t>(b)] = (n > 0) ? s / static_cast<double>(n) : 0.0;
    }
    return out;
}

// Min-max normalize a pooled vector into [0, 1]. Spike encoders (poisson,
// latency) interpret each entry as a firing probability / spike-time fraction,
// so raw amplitudes (EEG µV, wav in [-1, 1]) must be scaled first — otherwise
// almost no encoder neuron reaches the LIF threshold and the SNN latent is all
// zeros. Constant signals map to 0.
std::vector<float> normalize01(const std::vector<double>& v)
{
    std::vector<float> out(v.size(), 0.0f);
    if (v.empty()) return out;
    const auto [lo_it, hi_it] = std::minmax_element(v.begin(), v.end());
    const double lo = *lo_it;
    const double range = std::max(*hi_it - lo, 1e-12);
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>((v[i] - lo) / range);
    return out;
}

// One spike frame (1, D) for time step t under the chosen temporal code.
//   poisson — Bernoulli(value): rate code, value ≈ mean firing rate over T.
//   latency — spike from t_spike = round((1-value)(T-1)) onward: stronger inputs
//             fire earlier (matches the Experiment04 latency encoder).
//   direct  — pass the analog value through unchanged (non-spiking; used for
//             ANN-AE and as an SNN fallback).
nn::Tensor spike_frame(const std::vector<float>& norm,
    const std::string& encoding,
    int t,
    int time_steps,
    std::mt19937& rng)
{
    const auto D = static_cast<nn::Index>(norm.size());
    nn::Tensor frame(1, D);
    if (encoding == "poisson")
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (nn::Index d = 0; d < D; ++d)
            frame.at(0, d) = (dist(rng) < norm[static_cast<size_t>(d)]) ? 1.0f : 0.0f;
    }
    else if (encoding == "latency")
    {
        const int last = std::max(1, time_steps - 1);
        for (nn::Index d = 0; d < D; ++d)
        {
            const int t_spike =
                static_cast<int>(std::lround((1.0f - norm[static_cast<size_t>(d)]) * last));
            frame.at(0, d) = (t >= t_spike) ? 1.0f : 0.0f;
        }
    }
    else // "direct"
    {
        for (nn::Index d = 0; d < D; ++d) frame.at(0, d) = norm[static_cast<size_t>(d)];
    }
    return frame;
}

// Detects the one failure mode a spike loss can hit SILENTLY: an all-zero gradient.
//
// SpikeTimeLossImpl::backward writes a gradient only at the predicted first-spike row
// (`if (t < T) grad.at(t*B + b, f) = g;`). A unit that never crosses the 0.5 spike
// threshold has pt == T, the guard fails, and NO gradient is written for it. If that
// holds for every unit in every batch, training runs to completion, reports a loss,
// and changes nothing — a "converged" result that never learned. Neither the Trainer
// nor the loss reports this today.
//
// This wrapper forwards the whole LossType contract to `inner` and counts how many
// backward() calls produced an all-zero gradient. Stats live behind a shared_ptr so
// the caller can still read them after the Trainer has taken its own copy.
template <typename Inner>
struct GradientLivenessGuard
{
    using Tensor = nn::Tensor;

    struct Stats
    {
        long backward_calls = 0;
        long zero_grad_calls = 0;
    };

    Inner inner;
    std::shared_ptr<Stats> stats = std::make_shared<Stats>();

    GradientLivenessGuard() = default;
    explicit GradientLivenessGuard(Inner in) : inner(std::move(in)) {}

    void train(bool on)
    {
        inner.train(on);
    }
    void set_target(const Tensor& t)
    {
        inner.set_target(t);
    }
    auto forward(const Tensor& x, bool requires_grad = true) -> Tensor
    {
        return inner.forward(x, requires_grad);
    }
    auto backward(const Tensor& x) -> Tensor
    {
        Tensor g = inner.backward(x);
        ++stats->backward_calls;
        float mag = 0.0f;
        for (size_t i = 0; i < g.rows() && mag == 0.0f; ++i)
            for (size_t j = 0; j < g.cols(); ++j)
                if (g.at(i, j) != 0.0f)
                {
                    mag = 1.0f;
                    break;
                }
        if (mag == 0.0f) ++stats->zero_grad_calls;
        return g;
    }
    // Expose the SNN energy diagnostic ONLY when the wrapped loss actually has one.
    // Defining it unconditionally would make Trainer's has_last_mean_rate<> detector
    // true for losses that have no rate, forcing a fabricated value — a fallback.
    // Constrained so the member does not exist when Inner lacks it.
    [[nodiscard]] float last_mean_rate() const
        requires requires(const Inner& i) { i.last_mean_rate(); }
    {
        return inner.last_mean_rate();
    }
};

/// The seed for one spike frame, from (experiment seed, sample, step).
///
/// Pure in those three, and -- the part that matters -- THE SAME function in
/// both places frames are drawn: once to train the autoencoder, once to
/// encode the features it produces. Draw them differently and the model is
/// asked to encode input it never saw. That does not fail; it returns worse
/// features, on a run that still looks reproducible because the weights were
/// seeded.
inline auto frame_seed(std::uint32_t seed, size_t sample, int step) -> std::uint32_t
{
    return seed + static_cast<std::uint32_t>(sample) * 1009u + static_cast<std::uint32_t>(step);
}

/// Pool every raw signal to the AE's input width, then min-max normalize it.
auto normalize_signals(const std::vector<std::vector<double>>& raw_signals)
    -> std::vector<std::vector<float>>
{
    std::vector<std::vector<float>> normed;
    normed.reserve(raw_signals.size());
    for (const auto& sig : raw_signals)
        normed.push_back(normalize01(pool_signal(sig, kAeInputFeatures)));
    return normed;
}

/// `time_count` spike frames per sample, contiguous, at index = s*T + t.
auto build_spike_frames(const std::vector<std::vector<float>>& normed,
    const std::string& encoding,
    int time_count,
    std::uint32_t seed) -> std::vector<nn::Tensor>
{
    std::vector<nn::Tensor> frames;
    frames.reserve(normed.size() * static_cast<size_t>(time_count));
    for (size_t s = 0; s < normed.size(); ++s)
        for (int t = 0; t < time_count; ++t)
        {
            std::mt19937 rng(frame_seed(seed, s, t));
            frames.push_back(spike_frame(normed[s], encoding, t, time_count, rng));
        }
    return frames;
}

/// Re-lay per-(sample, t) frames into the time-major groups the loss reads.
///
/// `LifBPTT` consumes a TIME-MAJOR `(T*B, F)` tensor and unrolls the membrane
/// internally; `SpikeTimeLossImpl` indexes rows as `t*B + b`. The Trainer
/// cannot build that layout: `create_batch()` stacks multi-row samples into a
/// 3-D `(B, T, C)` tensor -- wrong rank AND wrong row order -- and it
/// reshuffles indices every epoch, so consecutive samples are not a stable
/// group.
///
/// So each training "sample" is pre-interleaved here: one group of
/// `batch_size` inputs as `(T*g, D)` with row = `t*g + b`. Real batching
/// survives (g inputs per gradient step), and a trailing partial group is
/// fine because the loss derives B from `rows / T`.
auto to_time_major_groups(
    const std::vector<nn::Tensor>& frames, size_t sample_count, int time_count, int batch_size)
    -> std::vector<nn::Tensor>
{
    const int T = time_count;
    // Re-lay the per-(sample, t) frames above into time-major groups:
    // group k holds inputs [k*G, k*G+g) as a (T*g, D) tensor with row = t*g + b.
    const int G = std::max(1, batch_size);
    const auto D = static_cast<nn::Index>(kAeInputFeatures);
    std::vector<nn::Tensor> grouped;
    grouped.reserve((sample_count + static_cast<size_t>(G) - 1) / static_cast<size_t>(G));

    for (size_t base = 0; base < sample_count; base += static_cast<size_t>(G))
    {
        const size_t g = std::min<size_t>(static_cast<size_t>(G), sample_count - base);
        nn::Tensor group(static_cast<nn::Index>(static_cast<size_t>(T) * g), D);
        for (size_t b = 0; b < g; ++b)
            for (int t = 0; t < T; ++t)
            {
                // frames were emitted contiguously per sample: index = s*T + t
                const nn::Tensor& f =
                    frames[(base + b) * static_cast<size_t>(T) + static_cast<size_t>(t)];
                const auto row = static_cast<nn::Index>(static_cast<size_t>(t) * g + b); // t*g + b
                for (nn::Index d = 0; d < D; ++d) group.at(row, d) = f.at(0, d);
            }
        grouped.push_back(std::move(group));
    }
    return grouped;
}

/// Assemble the autoencoder's configuration from the profile's spec.
///
/// Every `throw` below replaces a silent default, and each default hid a
/// different lie: unreadable layer widths would have built a 64/16 network
/// the profile never asked for, and an empty loss token would have trained
/// with MSE while the profile said `mae`.
///
/// `time_steps` is the same trap without a throw: left at 1, `LifBPTT`
/// behaves exactly like a single-step `Lif` -- a model with no temporal
/// credit assignment that still trains and still reports a loss.
///
/// The initializer seed matters as much. Without it the initializers fall
/// back to `std::random_device`, so the SAME profile with the SAME seed
/// produced DIFFERENT features every run: only the frames were seeded, never
/// the weights. It was also a measured flake source --
/// `ThesisSnnAe.PoissonLatentIsNonDegenerate` failed 6 of 25 runs when an
/// unlucky draw left every encoder neuron below V_th.
auto build_ae_config(const ThesisConfig::AutoencoderConfig& spec,
    int time_count,
    std::uint32_t seed,
    float voltage_threshold,
    const std::string& ae_loss_type,
    bool spiking) -> AutoencoderConfig
{
    const int T = time_count;
    AutoencoderConfig ae_cfg;
    ae_cfg.input_features = kAeInputFeatures;
    // No silent architecture defaults: an unparseable spec must fail, not quietly
    // build a 64/16 network the profile never asked for.
    constexpr int kNoDim = -1;
    ae_cfg.hidden_size = first_encoder_dim(spec.encoder_layer_spec, kNoDim);
    ae_cfg.latent_size = last_encoder_dim(spec.encoder_layer_spec, kNoDim);
    if (ae_cfg.hidden_size == kNoDim || ae_cfg.latent_size == kNoDim)
        throw std::invalid_argument(
            "ThesisFeatureExtraction: could not read hidden/latent widths from "
            "autoencoder.encoder_layer_spec — every entry must be linear:<width>[:act]");
    ae_cfg.depth = std::max<int>(1, static_cast<int>(spec.encoder_layer_spec.size()) - 1);
    // Forward the FULL per-layer specs so the builder honours the exact widths of every
    // layer (build_{snn,ann}_encoder take the spec branch when these are non-empty).
    // Without this the builder falls back to a uniform-width taper derived from
    // hidden_size/depth, silently discarding the genome's per-layer neuron counts.
    ae_cfg.encoder_layer_spec = spec.encoder_layer_spec;
    ae_cfg.decoder_layer_spec = spec.decoder_layer_spec;
    if (ae_loss_type.empty())
        throw std::invalid_argument(
            "ThesisFeatureExtraction: ae_loss_type is empty — refusing to guess a "
            "reconstruction loss. Set autoencoder.ae_loss_type explicitly "
            "(mse|mae|spikecount|spiketime).");
    ae_cfg.loss_type = ae_loss_type;
    ae_cfg.delta_t = 1.0f;
    ae_cfg.voltage_threshold = voltage_threshold;
    // Sequence length for the BPTT unroll. Leaving this at its default of 1 would make
    // LifBPTT behave like a single-step Lif — a silent downgrade, not an error.
    ae_cfg.time_steps = spiking ? T : 1;
    ae_cfg.firing_rate_reg_lambda = spec.firing_rate_reg_lambda;
    ae_cfg.firing_rate_min = spec.firing_rate_min;
    ae_cfg.firing_rate_max = spec.firing_rate_max;
    // Seed weight initialisation from the experiment seed. Without this the initializers
    // fall back to std::random_device (see include/initializers/{xavier,kaiming_snn}.hpp),
    // so the SAME profile with the SAME seed produced DIFFERENT features on every run:
    // only the spike frames were seeded, never the weights. Two consequences this fixes:
    //   1. Autoencoder feature extraction is now reproducible, like the handcrafted path.
    //   2. It removes a real source of flakiness -- ThesisSnnAe.PoissonLatentIsNonDegenerate
    //      failed ~24% of runs (6/25 measured) because an unlucky draw left every encoder
    //      neuron below V_th, yielding an all-zero latent.
    // Guayaquil already did this (GuayaquilExperiment.cpp: snn_config.initializer_seed = run_seed);
    // Thesis was the odd one out.
    ae_cfg.initializer_seed = seed;
    return ae_cfg;
}

/// Training knobs, straight from the profile.
auto build_trainer_config(const ThesisConfig::Training& training, int batch_size, bool time_major)
    -> nn::training::TrainerConfig
{
    nn::training::TrainerConfig trainer_cfg;
    trainer_cfg.epochs = training.epochs;
    trainer_cfg.learning_rate = training.effective_learning_rate();
    trainer_cfg.optimizer_type = training.optimizer_type;
    trainer_cfg.optimizer_momentum = training.optimizer_momentum;
    trainer_cfg.grad_clip_norm = training.gradient_clip_norm;
    // For spiketime each training "sample" is ALREADY an interleaved (T*g, D) batch of
    // g inputs (see the grouping above), so the Trainer feeds exactly one group per
    // step. Real batch size is still `batch_size` — it just lives in the sample layout
    // instead of in create_batch(), which cannot produce the required time-major order.
    trainer_cfg.batch_size = time_major ? 1 : std::max(1, batch_size);

    return trainer_cfg;
}

/// Train the autoencoder with the loss the profile named.
///
/// The Trainer takes its loss as a COMPILE-TIME parameter
/// (`Trainer<ModelType, LossType>`), so a string from a JSON profile has to
/// be turned into a concrete instantiation somewhere, and this is that
/// somewhere. Before this dispatch existed the AE path was hard-wired to
/// `MSELossImpl`, and `mae` was simply unreachable from a thesis or GA
/// profile that asked for it -- the run trained, reported a loss, and used
/// the wrong objective.
///
/// An unknown token throws rather than falling back to MSE, for the same
/// reason.
template <typename AEType>
void train_autoencoder(AEType& model,
    const nn::training::TrainerConfig& trainer_cfg,
    const std::vector<nn::Tensor>& train_samples,
    const std::string& loss_token,
    const ThesisConfig::AutoencoderConfig& spec,
    int time_count,
    const std::string& label_suffix,
    const std::string& ae_kind,
    const std::string& encoding)
{
    const int T = time_count;
    // The Trainer's loss is a COMPILE-TIME template parameter
    // (Trainer<ModelType, LossType>), so the profile's ae_loss_type string has to be
    // dispatched to a concrete instantiation here. Before this, the AE path was hard-wired
    // to the default MSELossImpl and `mae` was unreachable from a thesis/GA profile.
    // Match the Guayaquil (Guayaquil) TUI: give the training bar a description + loss type so the
    // metadata line says WHAT is training, not just an anonymous "Autoencoder training".
    // No fold counter here — feature extraction trains one AE over the whole set (0,1 hides
    // the "run X/Y" column), so col3 shows just the loss, exactly like the Guayaquil bars.
    auto make_cb = [&](const char* tag)
    {
        auto cb =
            std::make_shared<nn::training::ProgressCallback>("Autoencoder training" + label_suffix);
        cb->set_metadata(ae_kind + " (" + encoding + ")", 0, 1, tag);
        return cb;
    };

    if (loss_token == "mae")
    {
        nn::training::Trainer<AEType, MAELossImpl<nn::Backend>> trainer(model, trainer_cfg);
        trainer.add_callback(make_cb("MAE"));
        (void) trainer.fit_autoencoder(train_samples);
    }
    else if (loss_token == "spikecount")
    {
        // Rate-coded (poisson) reconstruction: elementwise MSE over the spike tensor
        // plus a firing-rate band penalty. Shape-agnostic, so it works with the normal
        // (B, D) frame batching. Reuse the encoder's configured rate band so the loss
        // and the encoder regularizer pull toward the same target rate.
        SpikeCountLossImpl<nn::Backend> inner;
        inner.rate_reg_lambda = spec.firing_rate_reg_lambda;
        inner.min_rate = spec.firing_rate_min;
        inner.max_rate = spec.firing_rate_max;
        GradientLivenessGuard<SpikeCountLossImpl<nn::Backend>> loss(std::move(inner));
        auto stats = loss.stats;
        nn::training::Trainer<AEType, GradientLivenessGuard<SpikeCountLossImpl<nn::Backend>>>
            trainer(model, trainer_cfg, std::move(loss));
        trainer.add_callback(make_cb("SpikeCount"));
        (void) trainer.fit_autoencoder(train_samples);
        assert_gradients_were_live(stats->backward_calls,
            stats->zero_grad_calls,
            "spikecount",
            encoding,
            spec.firing_rate_reg_lambda);
    }
    else if (loss_token == "spiketime")
    {
        // Latency-coded reconstruction: MSE over first-spike times. Requires the
        // time-major (T*B, F) layout built above with batch_size forced to 1 (B == 1).
        SpikeTimeLossImpl<nn::Backend> inner(T);
        GradientLivenessGuard<SpikeTimeLossImpl<nn::Backend>> loss(std::move(inner));
        auto stats = loss.stats;
        nn::training::Trainer<AEType, GradientLivenessGuard<SpikeTimeLossImpl<nn::Backend>>>
            trainer(model, trainer_cfg, std::move(loss));
        trainer.add_callback(make_cb("SpikeTime"));
        (void) trainer.fit_autoencoder(train_samples);
        assert_gradients_were_live(stats->backward_calls,
            stats->zero_grad_calls,
            "spiketime",
            encoding,
            spec.firing_rate_reg_lambda);
    }
    else if (loss_token == "mse")
    {
        nn::training::Trainer<AEType, MSELossImpl<nn::Backend>> trainer(model, trainer_cfg);
        trainer.add_callback(make_cb("MSE"));
        (void) trainer.fit_autoencoder(train_samples);
    }
    else
    {
        // Unreachable via ThesisConfig::validate(), but an unknown token must never be
        // silently treated as MSE — that would train a different objective than asked.
        throw std::invalid_argument("ThesisFeatureExtraction: unsupported ae_loss_type \"" +
                                    loss_token + "\" (expected mse|mae|spikecount|spiketime)");
    }
}

/// The feature vector per sample: the MEAN latent over its T spike frames.
///
/// The membrane is reset ONCE per sample and then integrates across all T
/// frames. That is the whole point of a spiking encoder: a weak, individually
/// sub-threshold current accumulates until it fires. Reset between frames and
/// each step becomes an isolated stochastic threshold, which collapses the
/// latent to noise. With `T == 1` (the direct/ANN path) the distinction
/// disappears.
///
/// The frames are redrawn here rather than kept from training, and they are
/// identical because both sides go through `frame_seed`. That shared function
/// is load-bearing: encoding different frames than were trained on degrades
/// the features silently, with nothing in the run to show for it.
template <typename AEType>
auto encode_mean_latents(AEType& model,
    const std::vector<std::vector<float>>& normed,
    const std::string& encoding,
    int time_count,
    std::uint32_t seed,
    size_t latent_dim,
    bool time_major) -> std::vector<std::vector<double>>
{
    const int T = time_count;
    std::vector<std::vector<double>> vectors;
    vectors.reserve(normed.size());
    for (size_t s = 0; s < normed.size(); ++s)
    {
        std::vector<double> acc(latent_dim, 0.0);
        model.reset_state();

        if (time_major)
        {
            // One sample => B == 1, so the (T, D) sequence IS the (T*1, F) tensor
            // LifBPTT expects; it unrolls all T steps in a single encode() call.
            // Feeding frames one at a time would violate rows % time_steps == 0.
            const auto D = static_cast<nn::Index>(normed[s].size());
            nn::Tensor seq(static_cast<nn::Index>(T), D);
            for (int t = 0; t < T; ++t)
            {
                std::mt19937 rng(
                    seed + static_cast<std::uint32_t>(s) * 1009u + static_cast<std::uint32_t>(t));
                const nn::Tensor f = spike_frame(normed[s], encoding, t, T, rng);
                for (nn::Index d = 0; d < D; ++d) seq.at(static_cast<nn::Index>(t), d) = f.at(0, d);
            }
            const nn::Tensor latent = model.encode(seq, false); // (T, latent_dim)
            for (nn::Index t = 0; t < static_cast<nn::Index>(T); ++t)
                for (size_t k = 0; k < latent_dim && k < static_cast<size_t>(latent.cols()); ++k)
                    acc[k] += latent.at(t, static_cast<nn::Index>(k));
        }
        else
        {
            for (int t = 0; t < T; ++t)
            {
                std::mt19937 rng(
                    seed + static_cast<std::uint32_t>(s) * 1009u + static_cast<std::uint32_t>(t));
                const nn::Tensor frame = spike_frame(normed[s], encoding, t, T, rng);
                const auto lat = tensor_to_vec(model.encode(frame, false));
                for (size_t k = 0; k < latent_dim && k < lat.size(); ++k) acc[k] += lat[k];
            }
        }
        for (double& a : acc) a /= static_cast<double>(T);
        vectors.push_back(std::move(acc));
    }
    return vectors;
}

// Train a Protocol autoencoder (SNN-AE or ANN-AE) on pooled+normalized input
// vectors and return the latent (bottleneck) vector per sample. AEType is a
// Module with an AutoencoderConfig constructor and an encode() method
// (ProtocolAutoencoder or ProtocolSpikingAutoencoder).
//
// Temporal coding (SNN-AE): a LIF autoencoder only carries information through
// spikes, so each sample is expanded into `time_steps` spike frames via the
// `encoding` scheme. The AE is trained to reconstruct those frames, and the
// per-sample feature is the MEAN latent (spike rate) over the T frames. With
// `encoding == "direct"` (ANN-AE) the sample stays a single analog vector and
// T is ignored — the original one-shot behaviour, just min-max normalized.
template <typename AEType>
std::vector<std::vector<double>> run_protocol_ae(
    const std::vector<std::vector<double>>& raw_signals,
    const ThesisConfig::AutoencoderConfig& spec,
    const ThesisConfig::Training& training,
    const std::string& label_suffix,
    const std::string& ae_kind, // "SNN-AE" / "ANN-AE" — drives the TUI description
    int batch_size,
    const std::string& encoding,
    int time_steps,
    std::uint32_t seed,
    float voltage_threshold,
    const std::string& ae_loss_type,
    bool spiking,
    const std::function<void(const std::map<std::string, nn::Tensor>&)>& on_trained = nullptr)
{
    const bool temporal = (encoding != "direct");
    const int T = temporal ? std::max(1, time_steps) : 1;

    const std::vector<std::vector<float>> normed = normalize_signals(raw_signals);

    // `spiketime` is the one loss with a hard LAYOUT requirement: SpikeTimeLossImpl
    // indexes rows as `t*B + b`, i.e. a time-major (T*B, F) tensor. The default sample
    // shape here is a single (1, D) frame, which the Trainer stacks into (B, D) — batch
    // rows, no time axis — so feeding that to SpikeTimeLoss would silently reinterpret
    // unrelated samples as timesteps.
    //
    // The Trainer cannot build the required layout itself: create_batch() stacks
    // multi-row samples into a 3-D (B, T, C) tensor (wrong rank AND wrong row order),
    // and it reshuffles sample indices every epoch, so consecutive samples are not a
    // stable group. We therefore PRE-INTERLEAVE the batch ourselves: each training
    // "sample" is a whole group of `batch_size` inputs laid out as (T*g, D) with
    // row = t*g + b, and the Trainer runs one group per step. This keeps real batching
    // (g inputs per gradient step) while guaranteeing the exact layout the loss reads.
    // A trailing partial group is fine — the loss derives B = rows / T.
    // LifBPTT consumes a TIME-MAJOR (T*B, F) tensor and unrolls the membrane over
    // `time_steps` internally, so every spiking run must be laid out time-major — not
    // only `spiketime`. At T == 1 the grouping degenerates to ordinary (B, D) batching,
    // so this is uniform rather than special-cased.
    const bool time_major = spiking;

    std::vector<nn::Tensor> train_samples = build_spike_frames(normed, encoding, T, seed);
    if (time_major)
        train_samples = to_time_major_groups(train_samples, normed.size(), T, batch_size);

    const AutoencoderConfig ae_cfg =
        build_ae_config(spec, T, seed, voltage_threshold, ae_loss_type, spiking);

    AEType model(ae_cfg);

    const nn::training::TrainerConfig trainer_cfg =
        build_trainer_config(training, batch_size, time_major);

    train_autoencoder(
        model, trainer_cfg, train_samples, ae_loss_type, spec, T, label_suffix, ae_kind, encoding);

    // Weights are final the moment training returns — capture them here, before the
    // model goes out of scope at function exit (extract_features itself never keeps
    // them; see ModelSnapshotFn in ThesisFeatureExtraction.hpp).
    if (on_trained) on_trained(model.state_dict());

    const auto latent_dim = static_cast<size_t>(ae_cfg.latent_size);
    return encode_mean_latents(model, normed, encoding, T, seed, latent_dim, time_major);
}
} // namespace

} // namespace thesis
