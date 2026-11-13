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

#include <cmath>
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
#include "wave/audioFeatureExtraction.hpp" // rfft_power, hanning_window

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

// Raw-signal accessors, one per recorded modality. A sample without the requested
// signal is refused: the dataset loader keeps only trials with BOTH audio and EEG, so an
// empty tensor here is a loader bug. (These accessors used to substitute 256 zeros — a
// silent sample of silence that every extractor would then encode as a real trial.)
// Pre-emphasis is audio-only (see kPreEmphasisAlpha above).
using SignalGetter = std::function<std::vector<double>(const ThesisSample&)>;

inline void require_signal(const nn::Tensor& t, const char* what, const ThesisSample& sample)
{
    if (t.rows() > 0 && t.cols() > 0) return;
    throw std::runtime_error(std::string("ThesisFeatureExtraction: sample of subject ") +
                             std::to_string(sample.subject_id) + ", stimulus " +
                             std::to_string(sample.stimulus) + " has no " + what +
                             " signal. The loader pairs audio with EEG and must never emit "
                             "a sample missing either; check ThesisDataset.");
}

std::vector<double> voice_signal(const ThesisSample& sample)
{
    require_signal(sample.audio, "audio", sample);
    std::vector<double> sig = tensor_to_vec(sample.audio);
    apply_preemphasis(sig, kPreEmphasisAlpha);
    return sig;
}

std::vector<double> eeg_signal(const ThesisSample& sample)
{
    require_signal(sample.eeg, "EEG", sample);
    return tensor_to_vec(sample.eeg);
}

// Early fusion: one signal = voice samples followed by EEG samples, extracted
// through a single DTWPT/autoencoder pass (audit C12). The two halves have
// different native sample rates (Coretto: 44100 Hz voice vs 1024 Hz EEG; BIOMEX-DB:
// 16000 vs 128); the caller applies the voice rate to the whole concatenation for
// frequency-scale mapping since voice dominates the sample count (Coretto: 176400 vs
// 24576 per trial; BIOMEX-DB: 16000 vs 3584). This is an approximation, not a
// resampling — documented here so it isn't mistaken for a physically exact rate.
std::vector<double> fused_early_signal(const ThesisSample& sample)
{
    std::vector<double> sig = voice_signal(sample);
    const std::vector<double> eeg = eeg_signal(sample);
    sig.insert(sig.end(), eeg.begin(), eeg.end());
    return sig;
}

// Width of one encoder layer item "linear:<width>[:activation]". Anything else is refused:
// these helpers used to skip unreadable items and, when none was readable, return a
// caller-supplied width (64 hidden / 16 latent for the LSTM-AE) — a network the profile
// never described.
int linear_width(const std::string& item)
{
    const auto refuse = [&]()
    {
        return std::invalid_argument(
            "ThesisFeatureExtraction: autoencoder.encoder_layer_spec item \"" + item +
            "\" is not linear:<width>[:activation] with a positive integer width");
    };
    if (item.rfind("linear:", 0) != 0) throw refuse();
    const auto first_colon = item.find(':');
    const auto second_colon = item.find(':', first_colon + 1);
    const std::string token = item.substr(first_colon + 1, second_colon - first_colon - 1);
    if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos) throw refuse();
    const int width = std::stoi(token);
    if (width < 1) throw refuse();
    return width;
}

// Hidden width = the first encoder layer; latent width = the last one.
int first_encoder_dim(const std::vector<std::string>& spec)
{
    if (spec.empty())
        throw std::invalid_argument(
            "ThesisFeatureExtraction: autoencoder.encoder_layer_spec is empty");
    return linear_width(spec.front());
}

int last_encoder_dim(const std::vector<std::string>& spec)
{
    for (const auto& item : spec) (void) linear_width(item); // every item must be readable
    return linear_width(spec.back());
}

// The flat input width of the Protocol (SNN-AE / ANN-AE) autoencoders and of PCA: one
// vector of this many values per sample (not a sequence).
constexpr int kAeInputFeatures = 256;

// The LSTM-AE reads the same kAeInputFeatures values as a sequence: kLstmSteps steps of
// kLstmStepWidth adjacent spectrum bins, lowest frequencies first.
constexpr int kLstmSteps = 16;
constexpr int kLstmStepWidth = 16;
static_assert(kLstmSteps * kLstmStepWidth == kAeInputFeatures,
    "the LSTM-AE sequence must cover the autoencoders' input exactly");

// One spectral input row as a (steps x width) sequence: row t, column w = row[t*width + w].
// A row of another length is refused (it would be cut or padded silently).
nn::Tensor spectrum_to_sequence(const std::vector<float>& row, int steps, int width)
{
    if (row.size() != static_cast<size_t>(steps) * static_cast<size_t>(width))
        throw std::invalid_argument("ThesisFeatureExtraction: an LSTM-AE input of " +
                                    std::to_string(row.size()) + " values cannot be read as " +
                                    std::to_string(steps) + " steps of " + std::to_string(width));
    nn::Tensor t(static_cast<nn::Index>(steps), static_cast<nn::Index>(width));
    for (int f = 0; f < steps; ++f)
        for (int w = 0; w < width; ++w)
            t.at(static_cast<nn::Index>(f), static_cast<nn::Index>(w)) =
                row[static_cast<size_t>(f) * static_cast<size_t>(width) + static_cast<size_t>(w)];
    return t;
}

// Welch framing of that input: frames of 512 samples give 257 FFT bins, of which the 256
// above DC are kept; consecutive frames overlap by half.
constexpr int kSpectrumFrame = 2 * kAeInputFeatures;
constexpr int kSpectrumHop = kAeInputFeatures;

// What the autoencoders and PCA read: the log power spectrum of the signal, 256 values.
//
// WHY NOT THE WAVEFORM. The extractor must hand the classifier something that tells
// speakers apart, and a waveform does that through its frequency content, not through the
// value of any one sample. Until 2026-11-12 this input was the waveform averaged in 256
// blocks, and the averaging removed exactly that content:
//
//   signal                          one block         what survives the average
//   voice, 176400 samples (4 s)     689 samples       below ~32 Hz  (pitch is 85-255 Hz)
//   EEG, 6 x 4096 samples (4 s)      96 samples       below ~5 Hz   (alpha, beta gone)
//
// Measured on the 1974 trials (15 speakers, chance 0.067; linear classifier, 5 folds;
// scripts/analysis/thesis_ae_input_reference.py, scipy + PyTorch, none of this C++):
//
//   input, 256 values               voice    EEG
//   block-averaged waveform (old)   0.129    0.264
//   log power spectrum (this)       0.854    0.618
//
// The failure was SILENT: every autoencoder trained, every classifier converged on its
// training set, and only the test EER (0.48-0.52, chance) showed that the input held
// nothing. It read as "autoencoders and PCA are poor extractors" when it was the input.
//
// WHAT IT IS. Welch's estimate: cut the signal in half-overlapping frames of 512 samples,
// taper each with a Hann window, take the power of its FFT, average over the frames, drop
// the DC bin and take log10. A frame's position (its phase) no longer matters, so two
// utterances of one speaker give close vectors. The signal is the same one the handcrafted
// extractor reads (the SignalGetter's), so both families see the same data; samples after
// the last whole frame (at most 255) are not read.
//
// Nothing is substituted: a signal shorter than one frame, a non-finite sample and a bin
// of zero power (which has no logarithm) are refused.
std::vector<double> log_power_spectrum(const std::vector<double>& sig)
{
    const auto frame = static_cast<size_t>(kSpectrumFrame);
    const auto hop = static_cast<size_t>(kSpectrumHop);
    if (sig.size() < frame)
        throw std::invalid_argument("ThesisFeatureExtraction: a signal of " +
                                    std::to_string(sig.size()) +
                                    " samples has no power spectrum of " +
                                    std::to_string(kSpectrumFrame) + "-sample frames");
    for (double x : sig)
        if (!std::isfinite(x))
            throw std::invalid_argument("ThesisFeatureExtraction: non-finite value in a signal");

    const std::vector<double> window = nn::core::wave::hanning_window(kSpectrumFrame);
    const size_t n_frames = (sig.size() - frame) / hop + 1;
    std::vector<std::vector<float>> frames(n_frames, std::vector<float>(frame));
    for (size_t f = 0; f < n_frames; ++f)
        for (size_t i = 0; i < frame; ++i)
            frames[f][i] = static_cast<float>(sig[f * hop + i] * window[i]);

    // (n_frames, 257): bin 0 is DC, bins 1..256 go up to the Nyquist frequency.
    const nn::Tensor power = nn::core::wave::rfft_power(frames, kSpectrumFrame);

    std::vector<double> out(static_cast<size_t>(kAeInputFeatures), 0.0);
    for (int b = 0; b < kAeInputFeatures; ++b)
    {
        double sum = 0.0;
        for (size_t f = 0; f < n_frames; ++f)
            sum += static_cast<double>(power.at(static_cast<long>(f), b + 1));
        const double mean = sum / static_cast<double>(n_frames);
        if (!(mean > 0.0) || !std::isfinite(mean))
            throw std::invalid_argument("ThesisFeatureExtraction: frequency bin " +
                                        std::to_string(b + 1) + " has mean power " +
                                        std::to_string(mean) + ", which has no logarithm");
        out[static_cast<size_t>(b)] = std::log10(mean);
    }
    return out;
}

// Min-max normalize a log power spectrum into [0, 1]. Spike encoders (poisson,
// latency) interpret each entry as a firing probability / spike-time fraction,
// so log powers (any real number) must be scaled first — otherwise
// almost no encoder neuron reaches the LIF threshold and the SNN latent is all
// zeros. A constant or non-finite vector has no [0, 1] scaling and is refused: a
// constant one used to become all zeros (a flat-lined trial passed as a signal), and a
// NaN fell through min/max untouched.
std::vector<float> normalize01(const std::vector<double>& v)
{
    if (v.empty()) throw std::invalid_argument("ThesisFeatureExtraction: normalize01 of nothing");
    for (double x : v)
        if (!std::isfinite(x))
            throw std::invalid_argument("ThesisFeatureExtraction: non-finite value in a signal");
    const auto [lo_it, hi_it] = std::minmax_element(v.begin(), v.end());
    const double lo = *lo_it;
    const double range = *hi_it - lo;
    if (!(range > 0.0))
        throw std::invalid_argument("ThesisFeatureExtraction: flat spectrum (every bin is " +
                                    std::to_string(lo) + "); it has no [0, 1] scaling");
    std::vector<float> out(v.size(), 0.0f);
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>((v[i] - lo) / range);
    return out;
}

// One spike frame (1, D) for time step t under the chosen temporal code.
//   poisson — Bernoulli(value): rate code, value ≈ mean firing rate over T.
//   latency — spike from t_spike = round((1-value)(T-1)) onward: stronger inputs
//             fire earlier (matches the Experiment04 latency encoder).
//   direct  — pass the analog value through unchanged (non-spiking; the ANN-AE
//             and the direct-coded SNN-AE).
// Any other token is refused (it used to be treated as "direct").
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
        // With 1 step this used to clamp to max(1, 0) and spike half the inputs at t=0.
        if (time_steps < 2)
            throw std::invalid_argument(
                "ThesisFeatureExtraction: latency encoding needs time_steps >= 2, got " +
                std::to_string(time_steps));
        const int last = time_steps - 1;
        for (nn::Index d = 0; d < D; ++d)
        {
            const int t_spike =
                static_cast<int>(std::lround((1.0f - norm[static_cast<size_t>(d)]) * last));
            frame.at(0, d) = (t >= t_spike) ? 1.0f : 0.0f;
        }
    }
    else if (encoding == "direct")
    {
        for (nn::Index d = 0; d < D; ++d) frame.at(0, d) = norm[static_cast<size_t>(d)];
    }
    else
    {
        throw std::invalid_argument("ThesisFeatureExtraction: unknown encoding \"" + encoding +
                                    "\" (expected poisson, latency or direct)");
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

/// The autoencoders' and PCA's input: every signal's log power spectrum (kAeInputFeatures
/// values, see log_power_spectrum), min-max normalized per sample.
auto normalize_signals(const std::vector<std::vector<double>>& raw_signals)
    -> std::vector<std::vector<float>>
{
    std::vector<std::vector<float>> normed;
    normed.reserve(raw_signals.size());
    for (const auto& sig : raw_signals) normed.push_back(spectral_input(sig));
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
    if (batch_size < 1)
        throw std::invalid_argument(
            "ThesisFeatureExtraction: batch size must be >= 1, got " + std::to_string(batch_size));
    const int G = batch_size;
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
    // No silent architecture defaults: an unparseable spec throws (linear_width).
    ae_cfg.hidden_size = first_encoder_dim(spec.encoder_layer_spec);
    ae_cfg.latent_size = last_encoder_dim(spec.encoder_layer_spec);
    ae_cfg.depth = std::max<int>(1, static_cast<int>(spec.encoder_layer_spec->size()) - 1);
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
    // Firing-rate regularization exists only for the spiking encoder, and its band only
    // when it is on; the profile declares exactly those (ThesisConfig refuses the rest).
    // The non-spiking model has no firing rate: AutoencoderConfig's own fields stay unread.
    if (spiking)
    {
        ae_cfg.firing_rate_reg_lambda = spec.firing_rate_reg_lambda;
        if (spec.firing_rate_reg_lambda > 0.0f)
        {
            ae_cfg.firing_rate_min = spec.firing_rate_min;
            ae_cfg.firing_rate_max = spec.firing_rate_max;
        }
    }
    // Seed weight initialisation from the experiment seed. Without this the initializers
    // fall back to std::random_device (see include/initializers/{xavier,kaiming_snn}.hpp),
    // so the SAME profile with the SAME seed produced DIFFERENT features on every run:
    // only the spike frames were seeded, never the weights. Two consequences this fixes:
    //   1. Autoencoder feature extraction is now reproducible, like the handcrafted path.
    //   2. It removes a real source of flakiness -- ThesisSnnAe.PoissonLatentIsNonDegenerate
    //      failed ~24% of runs (6/25 measured) because an unlucky draw left every encoder
    //      neuron below V_th, yielding an all-zero latent.
    // Meeting01 already did this (Meeting01Experiment.cpp: snn_config.initializer_seed = run_seed);
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
    trainer_cfg.learning_rate = training.learning_rate;
    trainer_cfg.optimizer_type = training.optimizer_type;
    // Momentum exists for sgd only; the profile declares it exactly then.
    if (training.optimizer_type == "sgd")
        trainer_cfg.optimizer_momentum = training.optimizer_momentum;
    trainer_cfg.grad_clip_norm = training.gradient_clip_norm;
    // For spiketime each training "sample" is ALREADY an interleaved (T*g, D) batch of
    // g inputs (see the grouping above), so the Trainer feeds exactly one group per
    // step. Real batch size is still `batch_size` — it just lives in the sample layout
    // instead of in create_batch(), which cannot produce the required time-major order.
    if (batch_size < 1)
        throw std::invalid_argument(
            "ThesisFeatureExtraction: batch size must be >= 1, got " + std::to_string(batch_size));
    trainer_cfg.batch_size = time_major ? 1 : batch_size;

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
    // Match the Meeting01 (Meeting01) TUI: give the training bar a description + loss type so the
    // metadata line says WHAT is training, not just an anonymous "Autoencoder training".
    // No fold counter here — feature extraction trains one AE over the whole set (0,1 hides
    // the "run X/Y" column), so col3 shows just the loss, exactly like the Meeting01 bars.
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
        if (spec.firing_rate_reg_lambda > 0.0f) // the band exists only when the penalty does
        {
            inner.min_rate = spec.firing_rate_min;
            inner.max_rate = spec.firing_rate_max;
        }
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

// Train a Protocol autoencoder (SNN-AE or ANN-AE) on the normalized log power spectrum
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
    const std::function<void(const std::map<std::string, nn::Tensor>&)>& on_trained)
{
    // `time_steps` is the frame count the caller resolved: the profile's T for a temporal
    // code, 1 for the direct code (which has one analog frame by definition).
    const int T = time_steps;
    if (T < 1)
        throw std::invalid_argument(
            "ThesisFeatureExtraction: frame count must be >= 1, got " + std::to_string(T));

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
