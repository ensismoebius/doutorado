#include "ThesisFeatureExtraction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "autoencoder/ProtocolAutoencoder.hpp"        // ANN-AE (non-spiking)
#include "autoencoder/ProtocolSpikingAutoencoder.hpp" // SNN-AE (spiking)
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "layers/losses/MAELoss.hpp"        // MAE reconstruction loss (Trainer.hpp only pulls MSE)
#include "layers/losses/SpikeCountLoss.hpp" // rate-coded (poisson) reconstruction
#include "layers/losses/SpikeTimeLoss.hpp"  // latency-coded reconstruction
#include "models/lstm/LSTMAutoencoder.hpp"
#include "progress/ProgressManager.hpp"
#include "training/ProgressCallback.hpp"
#include "wavelet/Types.hpp"
#include "wavelet/WaveletTransformResults.hpp"
#include "wavelet/waveletOperations.hpp"

namespace thesis
{

// Fail loudly when a spike loss contributed nothing. A run whose every gradient was
// zero must not be reported as a result — that is the silent-no-learning mode.
void assert_gradients_were_live(long backward_calls,
    long zero_grad_calls,
    const std::string& loss_token,
    const std::string& encoding,
    float firing_rate_reg_lambda)
{
    if (backward_calls == 0 || zero_grad_calls < backward_calls) return;

    throw std::runtime_error(
        "ThesisFeatureExtraction: ae_loss_type=" + loss_token + " (encoding=" + encoding +
        ") produced an ALL-ZERO gradient on every one of the " + std::to_string(backward_calls) +
        " training batches — the autoencoder trained on nothing and its features are "
        "meaningless. Cause: no output unit ever crossed the spike threshold, so the "
        "straight-through estimator had no spike time to attach a gradient to (the "
        "no-spike deadlock in .wiki/Concepts/Spike-Encoding.md). Fixes, in order of "
        "preference: (1) raise autoencoder.firing_rate_reg_lambda (currently " +
        std::to_string(firing_rate_reg_lambda) +
        ") so the encoder gets rate-derived gradient independent of spike position; "
        "(2) lower autoencoder.voltage_threshold so units fire at all; (3) enable tdBN "
        "upstream. Refusing to emit features from an untrained model.");
}

// ─── scalar descriptors ────────────────────────────────────────────────────

auto compute_zcr(const std::vector<double>& signal) -> double
{
    if (signal.size() < 2) return 0.0;
    int crossings = 0;
    for (size_t i = 1; i < signal.size(); ++i)
    {
        if ((signal[i] >= 0.0) != (signal[i - 1] >= 0.0)) ++crossings;
    }
    return static_cast<double>(crossings) / static_cast<double>(signal.size() - 1);
}

auto compute_entropy(const std::vector<double>& signal) -> double
{
    if (signal.empty()) return 0.0;
    double total = 0.0;
    for (double v : signal) total += std::abs(v);
    if (total == 0.0) return 0.0;

    double ent = 0.0;
    for (double v : signal)
    {
        double p = std::abs(v) / total;
        if (p > 0.0) ent -= p * std::log2(p);
    }
    return ent;
}

auto compute_teager(const std::vector<double>& signal) -> double
{
    if (signal.size() < 3) return 0.0;
    double sum = 0.0;
    for (size_t i = 1; i + 1 < signal.size(); ++i)
        sum += signal[i] * signal[i] - signal[i - 1] * signal[i + 1];
    return sum / static_cast<double>(signal.size() - 2);
}

auto compute_energy(const std::vector<double>& subband) -> double
{
    double e = 0.0;
    for (double v : subband) e += v * v;
    return e;
}

auto compute_jitter(const std::vector<double>& signal, double sample_rate) -> double
{
    std::vector<size_t> peaks;
    for (size_t i = 1; i + 1 < signal.size(); ++i)
    {
        if (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1] && signal[i] > 0.0)
            peaks.push_back(i);
    }
    if (peaks.size() < 3) return std::numeric_limits<double>::quiet_NaN();

    double period_sum = 0.0;
    double diff_sum = 0.0;
    for (size_t k = 1; k < peaks.size(); ++k)
    {
        double T = static_cast<double>(peaks[k] - peaks[k - 1]) / sample_rate;
        period_sum += T;
        if (k > 1)
        {
            double prev_T = static_cast<double>(peaks[k - 1] - peaks[k - 2]) / sample_rate;
            diff_sum += std::abs(T - prev_T);
        }
    }
    double mean_T = period_sum / static_cast<double>(peaks.size() - 1);
    if (mean_T == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return diff_sum / (static_cast<double>(peaks.size() - 2) * mean_T);
}

auto compute_shimmer(const std::vector<double>& signal, double /*sample_rate*/) -> double
{
    std::vector<size_t> peaks;
    for (size_t i = 1; i + 1 < signal.size(); ++i)
    {
        if (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1] && signal[i] > 0.0)
            peaks.push_back(i);
    }
    if (peaks.size() < 3) return std::numeric_limits<double>::quiet_NaN();

    double amp_sum = 0.0;
    double diff_sum = 0.0;
    for (size_t k = 0; k < peaks.size(); ++k)
    {
        amp_sum += signal[peaks[k]];
        if (k > 0) diff_sum += std::abs(signal[peaks[k]] - signal[peaks[k - 1]]);
    }
    double mean_A = amp_sum / static_cast<double>(peaks.size());
    if (mean_A == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return diff_sum / (static_cast<double>(peaks.size() - 1) * mean_A);
}

void apply_preemphasis(std::vector<double>& signal, double alpha)
{
    for (size_t n = signal.size(); n-- > 1;) signal[n] -= alpha * signal[n - 1];
}

// ─── frequency scale helpers ────────────────────────────────────────────────

namespace
{

// Zwicker & Terhardt (1980) Bark approximation.
double hz_to_bark(double f)
{
    return 13.0 * std::atan(0.00076 * f) + 3.5 * std::atan(std::pow(f / 7500.0, 2.0));
}

// O'Shaughnessy MEL formula.
double hz_to_mel(double f)
{
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

// Runtime name → mother-wavelet decomposition filter. Coefficient arrays have
// static storage (constexpr in Types.hpp), so returning spans over them is safe.
// Only the tags listed here (all with WaveletTraits specializations) are valid;
// ThesisConfig::validate() rejects any other name before extraction runs.
std::span<const double> wavelet_filter(const std::string& name)
{
    using namespace wavelets;
    static const std::unordered_map<std::string, std::span<const double>> table = {
        {"haar", get_wavelet<Haar>()},
        {"daub4", get_wavelet<Daub4>()},
        {"daub6", get_wavelet<Daub6>()},
        {"daub8", get_wavelet<Daub8>()},
        {"daub10", get_wavelet<Daub10>()},
        {"daub12", get_wavelet<Daub12>()},
        {"daub14", get_wavelet<Daub14>()},
        {"daub16", get_wavelet<Daub16>()},
        {"daub18", get_wavelet<Daub18>()},
        {"daub20", get_wavelet<Daub20>()},
        {"daub22", get_wavelet<Daub22>()},
        {"daub24", get_wavelet<Daub24>()},
        {"daub26", get_wavelet<Daub26>()},
        {"daub28", get_wavelet<Daub28>()},
        {"daub30", get_wavelet<Daub30>()},
        {"daub32", get_wavelet<Daub32>()},
        {"daub34", get_wavelet<Daub34>()},
        {"daub36", get_wavelet<Daub36>()},
        {"daub38", get_wavelet<Daub38>()},
        {"daub40", get_wavelet<Daub40>()},
        {"daub42", get_wavelet<Daub42>()},
        {"daub44", get_wavelet<Daub44>()},
        {"daub46", get_wavelet<Daub46>()},
    };
    const auto it = table.find(name);
    if (it == table.end())
        throw std::invalid_argument("ThesisFeatureExtraction: unknown wavelet \"" + name + "\"");
    return it->second;
}

// DCT-II of a real vector: C[k] = sum_n x[n] cos(pi/N (n+0.5) k), k=0..N-1.
// Used for the Category-2 cepstral transform (log energies -> cepstral coeffs).
std::vector<double> dct_ii(const std::vector<double>& x)
{
    const size_t N = x.size();
    std::vector<double> c(N, 0.0);
    if (N == 0) return c;
    for (size_t k = 0; k < N; ++k)
    {
        double acc = 0.0;
        for (size_t n = 0; n < N; ++n)
            acc += x[n] * std::cos(M_PI / static_cast<double>(N) * (static_cast<double>(n) + 0.5) *
                                   static_cast<double>(k));
        c[k] = acc;
    }
    return c;
}

// Group DTWPT sub-bands by perceptual frequency scale.
//
// "lfcc" → each sub-band is its own group (uniform linear spacing).
// "bark" → 24 Bark groups.
// "mel"  → 20 MEL groups.
//
// For Bark/MEL: each sub-band's center frequency is mapped to the scale;
// sub-bands sharing the same scale bin have their coefficients concatenated
// into a single group. Empty bins (no sub-bands mapped to them) are dropped.
//
// Returns a vector of groups; each group is the concatenated coefficients
// of one frequency bin — ready for descriptor computation.
std::vector<std::vector<double>> group_by_scale(wavelets::WaveletTransformResults& result,
    long n_parts,
    const std::string& scale,
    double sample_rate)
{
    if (scale == "lfcc")
    {
        // Linear spacing: one group per DTWPT sub-band.
        std::vector<std::vector<double>> groups;
        groups.reserve(static_cast<size_t>(n_parts));
        for (long p = 0; p < n_parts; ++p)
        {
            groups.push_back(wavelets::WaveletTransformResults::get_wavelet_packet_transforms(
                result.transformedSignal, p, result.levelsOfTransformation));
        }
        return groups;
    }

    const int n_bands = (scale == "bark") ? 24 : 20;
    const double nyquist = sample_rate / 2.0;
    const double max_sv = (scale == "bark") ? hz_to_bark(nyquist) : hz_to_mel(nyquist);

    std::vector<std::vector<double>> groups(static_cast<size_t>(n_bands));

    for (long p = 0; p < n_parts; ++p)
    {
        // Center frequency of sub-band p (uniform partition of Nyquist).
        double center_hz = (p + 0.5) * nyquist / static_cast<double>(n_parts);
        double sv = (scale == "bark") ? hz_to_bark(center_hz) : hz_to_mel(center_hz);
        int band = static_cast<int>(sv / max_sv * n_bands);
        band = std::clamp(band, 0, n_bands - 1);

        auto coefs = wavelets::WaveletTransformResults::get_wavelet_packet_transforms(
            result.transformedSignal, p, result.levelsOfTransformation);
        auto& g = groups[static_cast<size_t>(band)];
        g.insert(g.end(), coefs.begin(), coefs.end());
    }

    // Drop empty bins — may occur when n_parts < n_bands (low DTWPT levels).
    groups.erase(
        std::remove_if(
            groups.begin(), groups.end(), [](const std::vector<double>& g) { return g.empty(); }),
        groups.end());

    return groups;
}

} // namespace

// ─── handcrafted extraction ─────────────────────────────────────────────────

auto extract_handcrafted(const std::vector<double>& signal,
    const ThesisConfig::HandcraftedConfig& cfg,
    double sample_rate) -> std::vector<double>
{
    using wavelets::PACKET_WAVELET;

    const std::span<const double> filter = wavelet_filter(cfg.wavelet);

    auto result =
        wavelets::malat(signal, filter, PACKET_WAVELET, static_cast<unsigned int>(cfg.dtwpt_level));

    long n_parts = result.get_wavelet_packet_amount_of_parts();

    // Group sub-bands according to the perceptual frequency scale.
    auto groups = group_by_scale(result, n_parts, cfg.scale, sample_rate);

    const auto& descs = cfg.descriptors;
    bool want_energy = std::find(descs.begin(), descs.end(), "energy") != descs.end();
    bool want_zcr = std::find(descs.begin(), descs.end(), "zcr") != descs.end();
    bool want_entropy = std::find(descs.begin(), descs.end(), "entropy") != descs.end();
    bool want_teager = std::find(descs.begin(), descs.end(), "teager") != descs.end();
    bool want_jitter = std::find(descs.begin(), descs.end(), "jitter") != descs.end();
    bool want_shimmer = std::find(descs.begin(), descs.end(), "shimmer") != descs.end();

    std::vector<double> features;

    // Category 2: log + DCT-II over the band energies → cepstral coefficients
    // (LFCC/MFCC/BFCC by scale). Replaces the raw per-band energy descriptor.
    if (cfg.cepstral)
    {
        std::vector<double> log_energies;
        log_energies.reserve(groups.size());
        for (const auto& group : groups)
            log_energies.push_back(std::log(compute_energy(group) + 1e-10));
        const auto cepstral = dct_ii(log_energies);
        features.insert(features.end(), cepstral.begin(), cepstral.end());
    }

    for (const auto& group : groups)
    {
        if (want_energy && !cfg.cepstral) features.push_back(compute_energy(group));
        if (want_zcr) features.push_back(compute_zcr(group));
        if (want_entropy) features.push_back(compute_entropy(group));
        if (want_teager) features.push_back(compute_teager(group));
        if (want_jitter)
        {
            double j = compute_jitter(group, sample_rate);
            features.push_back(std::isnan(j) ? 0.0 : j);
        }
        if (want_shimmer)
        {
            double s = compute_shimmer(group, sample_rate);
            features.push_back(std::isnan(s) ? 0.0 : s);
        }
    }

    return features;
}

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

    // Normalize every sample once; spike frames are drawn from these.
    std::vector<std::vector<float>> normed;
    normed.reserve(raw_signals.size());
    for (const auto& sig : raw_signals)
        normed.push_back(normalize01(pool_signal(sig, kAeInputFeatures)));

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

    // Build the training set: T spike frames per sample (temporal), or one
    // analog vector per sample (direct). Seed is derived per (sample, step) so
    // the frames are reproducible for a given experiment seed.
    std::vector<nn::Tensor> train_samples;
    train_samples.reserve(normed.size() * static_cast<size_t>(T));
    for (size_t s = 0; s < normed.size(); ++s)
    {
        for (int t = 0; t < T; ++t)
        {
            std::mt19937 rng(
                seed + static_cast<std::uint32_t>(s) * 1009u + static_cast<std::uint32_t>(t));
            train_samples.push_back(spike_frame(normed[s], encoding, t, T, rng));
        }
    }

    if (time_major)
    {
        // Re-lay the per-(sample, t) frames above into time-major groups:
        // group k holds inputs [k*G, k*G+g) as a (T*g, D) tensor with row = t*g + b.
        const int G = std::max(1, batch_size);
        const auto D = static_cast<nn::Index>(kAeInputFeatures);
        std::vector<nn::Tensor> grouped;
        grouped.reserve((normed.size() + static_cast<size_t>(G) - 1) / static_cast<size_t>(G));

        for (size_t base = 0; base < normed.size(); base += static_cast<size_t>(G))
        {
            const size_t g = std::min<size_t>(static_cast<size_t>(G), normed.size() - base);
            nn::Tensor group(static_cast<nn::Index>(static_cast<size_t>(T) * g), D);
            for (size_t b = 0; b < g; ++b)
                for (int t = 0; t < T; ++t)
                {
                    // frames were emitted contiguously per sample: index = s*T + t
                    const nn::Tensor& f =
                        train_samples[(base + b) * static_cast<size_t>(T) + static_cast<size_t>(t)];
                    const auto row =
                        static_cast<nn::Index>(static_cast<size_t>(t) * g + b); // t*g + b
                    for (nn::Index d = 0; d < D; ++d) group.at(row, d) = f.at(0, d);
                }
            grouped.push_back(std::move(group));
        }
        train_samples = std::move(grouped);
    }

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

    AEType model(ae_cfg);

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

    // The Trainer's loss is a COMPILE-TIME template parameter
    // (Trainer<ModelType, LossType>), so the profile's ae_loss_type string has to be
    // dispatched to a concrete instantiation here. Before this, the AE path was hard-wired
    // to the default MSELossImpl and `mae` was unreachable from a thesis/GA profile.
    // Match the Guayaquil (Guayaquil) TUI: give the training bar a description + loss type so the
    // metadata line says WHAT is training, not just an anonymous "Autoencoder training".
    // No fold counter here — feature extraction trains one AE over the whole set (0,1 hides
    // the "run X/Y" column), so col3 shows just the loss, exactly like the Guayaquil bars.
    const std::string& loss_token = ae_loss_type;
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

    // Weights are final the moment training returns — capture them here, before the
    // model goes out of scope at function exit (extract_features itself never keeps
    // them; see ModelSnapshotFn in ThesisFeatureExtraction.hpp).
    if (on_trained) on_trained(model.state_dict());

    // Feature per sample = mean latent over its T spike frames. The membrane
    // state is reset ONCE per sample and then integrates across the T frames
    // (no reset between frames): this is the temporal integration that lets a
    // weak, sub-threshold per-step current accumulate into spikes — the whole
    // point of an SNN. Resetting every frame would make each step an isolated
    // stochastic threshold and collapse the latent. Direct (T==1) is unchanged.
    const auto latent_dim = static_cast<size_t>(ae_cfg.latent_size);
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
} // namespace

namespace
{

// Runs the configured strategy (handcrafted or autoencoder) against whatever
// single signal get_signal() returns per sample. Used directly for "voice"
// and "eeg" modality, and reused twice by late fusion (once per signal) and
// once by early fusion (on the pre-concatenated signal). label_suffix tags
// the resulting FeatureSet so voice-part/EEG-part/fused variants stay
// distinguishable in results output.
auto extract_features_core(const ThesisDatasetView& view,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const SignalGetter& get_signal,
    double sample_rate,
    const std::string& label_suffix,
    std::uint32_t seed,
    const std::string& part_tag = "",
    const ModelSnapshotFn& on_model_trained = nullptr) -> std::vector<FeatureSet>
{
    std::vector<FeatureSet> result;

    if (cfg.strategy == "handcrafted")
    {
        FeatureSet fs;
        fs.label = "handcrafted-" + cfg.handcrafted.scale + label_suffix;
        fs.vectors.reserve(view.samples.size());

        const auto n_samples = static_cast<long>(view.samples.size());
        uint32_t feat_bar = nn::progress::ProgressManager::instance().create_bar(
            "Feature extraction", static_cast<float>(n_samples));
        nn::progress::ProgressManager::instance().set_description(feat_bar,
            "DTWPT | scale=" + cfg.handcrafted.scale +
                "  descriptors=" + std::to_string(cfg.handcrafted.descriptors.size()));

        fs.vectors.resize(static_cast<size_t>(n_samples));
        long feat_done = 0;

#pragma omp parallel for schedule(dynamic, 4) shared(feat_done)
        for (long i = 0; i < n_samples; ++i)
        {
            const auto& sample = view.samples[static_cast<size_t>(i)];
            std::vector<double> sig = get_signal(sample);

            // Zero-pad to next power of two for DTWPT.
            size_t n = 1;
            while (n < sig.size()) n <<= 1;
            sig.resize(n, 0.0);

            fs.vectors[static_cast<size_t>(i)] =
                extract_handcrafted(sig, cfg.handcrafted, sample_rate);

            long done = 0;
#pragma omp atomic capture
            done = ++feat_done;
            nn::progress::ProgressManager::instance().update_bar(
                feat_bar, static_cast<float>(done));
        }
        nn::progress::ProgressManager::instance().complete_bar(feat_bar);
        result.push_back(std::move(fs));
    }
    else if (cfg.strategy == "autoencoder")
    {
        const std::string& ae_model = cfg.autoencoder.model;

        std::vector<std::vector<double>> raw_signals;
        raw_signals.reserve(view.samples.size());
        size_t max_len = 0;
        for (const auto& sample : view.samples)
        {
            std::vector<double> sig = get_signal(sample);
            max_len = std::max(max_len, sig.size());
            raw_signals.push_back(std::move(sig));
        }
        if (max_len == 0)
            throw std::runtime_error(
                "ThesisFeatureExtraction: no valid raw signals for autoencoder");

        FeatureSet fs;

        if (ae_model == "snn-ae")
        {
            fs.label = "autoencoder-snn" + label_suffix;
            // Spiking AE trains batched: row-vector samples stack into a 2D (B, F)
            // batch (Trainer::create_batch), and Lif/LifIntegrator resize their
            // membrane state to the batch shape. Same batch size as ANN-AE.
            fs.vectors = run_protocol_ae<ProtocolSpikingAutoencoder>(raw_signals,
                cfg.autoencoder,
                training,
                label_suffix,
                "SNN-AE",
                training.samples_per_batch,
                cfg.autoencoder.encoding,
                cfg.autoencoder.time_steps,
                seed,
                cfg.autoencoder.voltage_threshold,
                cfg.autoencoder.ae_loss_type,
                /*spiking=*/true,
                on_model_trained ? std::function<void(const std::map<std::string, nn::Tensor>&)>(
                                       [&](const std::map<std::string, nn::Tensor>& sd)
                                       { on_model_trained(part_tag, sd); })
                                 : nullptr);
        }
        else if (ae_model == "ann-ae")
        {
            fs.label = "autoencoder-ann" + label_suffix;
            // ANN-AE is non-spiking: always analog ("direct"), single frame.
            fs.vectors = run_protocol_ae<ProtocolAutoencoder>(raw_signals,
                cfg.autoencoder,
                training,
                label_suffix,
                "ANN-AE",
                training.samples_per_batch,
                "direct",
                1,
                seed,
                1.0f,
                cfg.autoencoder.ae_loss_type,
                /*spiking=*/false,
                on_model_trained ? std::function<void(const std::map<std::string, nn::Tensor>&)>(
                                       [&](const std::map<std::string, nn::Tensor>& sd)
                                       { on_model_trained(part_tag, sd); })
                                 : nullptr);
        }
        else // "lstm-ae" — sequence AE on windowed frames (Guayaquil-paper extractor)
        {
            const int frame_len =
                std::max<int>(1, static_cast<int>((max_len + kAeMaxFrames - 1) / kAeMaxFrames));
            const int T_frames = kAeMaxFrames;
            const size_t padded = static_cast<size_t>(T_frames) * static_cast<size_t>(frame_len);

            std::vector<nn::Tensor> train_samples;
            train_samples.reserve(raw_signals.size());
            for (auto& sig : raw_signals)
            {
                sig.resize(padded, 0.0);
                train_samples.push_back(vec_to_frame_tensor(sig, T_frames, frame_len));
            }

            nn::models::lstm::LSTMAutoencoderConfig ae_cfg;
            ae_cfg.input_size = frame_len;
            ae_cfg.seq_len = T_frames;
            ae_cfg.hidden_size = first_encoder_dim(cfg.autoencoder.encoder_layer_spec, 64);
            ae_cfg.latent_size = last_encoder_dim(cfg.autoencoder.encoder_layer_spec, 16);
            ae_cfg.num_layers =
                std::max<int>(1, static_cast<int>(cfg.autoencoder.encoder_layer_spec.size() / 2));

            nn::models::lstm::LSTMAutoencoder model(ae_cfg);

            nn::training::TrainerConfig trainer_cfg;
            trainer_cfg.epochs = training.epochs;
            trainer_cfg.learning_rate = training.effective_learning_rate();
            trainer_cfg.optimizer_type = training.optimizer_type;
            trainer_cfg.optimizer_momentum = training.optimizer_momentum;
            trainer_cfg.grad_clip_norm = training.gradient_clip_norm;
            trainer_cfg.batch_size = training.samples_per_batch;

            nn::training::Trainer<nn::models::lstm::LSTMAutoencoder> trainer(model, trainer_cfg);
            // Same enrichment as the SNN/ANN AE path: label the bar with the model + loss so
            // the metadata line matches the Guayaquil TUI look.
            auto ae_cb = std::make_shared<nn::training::ProgressCallback>(
                "Autoencoder training" + label_suffix);
            ae_cb->set_metadata("LSTM-AE", 0, 1, "MSE");
            trainer.add_callback(ae_cb);
            (void) trainer.fit_autoencoder(train_samples);

            fs.label = "autoencoder-lstm" + label_suffix;
            fs.vectors.reserve(train_samples.size());
            for (const auto& sample : train_samples)
                fs.vectors.push_back(tensor_to_vec(model.encode(sample, false)));
        }

        result.push_back(std::move(fs));
    }
    else
    {
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown strategy \"" + cfg.strategy + "\"");
    }

    return result;
}

} // namespace

auto extract_features(const ThesisDatasetView& view,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const std::string& modality,
    const std::string& fusion_mode,
    std::uint32_t seed,
    const ModelSnapshotFn& on_model_trained) -> std::vector<FeatureSet>
{
    if (modality == "voice")
        return extract_features_core(
            view, cfg, training, voice_signal, kVoiceSampleRate, "", seed, "", on_model_trained);

    if (modality == "eeg")
        return extract_features_core(
            view, cfg, training, eeg_signal, kEegSampleRate, "", seed, "", on_model_trained);

    if (modality != "fused")
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown modality \"" + modality + "\"");

    if (fusion_mode == "early")
    {
        return extract_features_core(view,
            cfg,
            training,
            fused_early_signal,
            kVoiceSampleRate,
            "-fused-early",
            seed,
            "",
            on_model_trained);
    }

    if (fusion_mode != "late")
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown fusion_mode \"" + fusion_mode + "\"");

    // Late fusion: extract independently per signal, then concatenate the
    // resulting feature vectors sample-by-sample (audit C12). Two INDEPENDENT models
    // train here (voice AE + EEG AE) — tag each snapshot so a caller merging them
    // (e.g. paraconsistentGA's weight-snapshot capture) can tell them apart.
    auto voice_sets = extract_features_core(
        view, cfg, training, voice_signal, kVoiceSampleRate, "", seed, "voice", on_model_trained);
    auto eeg_sets = extract_features_core(
        view, cfg, training, eeg_signal, kEegSampleRate, "", seed, "eeg", on_model_trained);

    if (voice_sets.size() != eeg_sets.size())
        throw std::runtime_error(
            "ThesisFeatureExtraction: late fusion produced mismatched FeatureSet counts");

    std::vector<FeatureSet> result;
    result.reserve(voice_sets.size());
    for (size_t k = 0; k < voice_sets.size(); ++k)
    {
        auto& vfs = voice_sets[k];
        auto& efs = eeg_sets[k];
        if (vfs.vectors.size() != efs.vectors.size())
            throw std::runtime_error(
                "ThesisFeatureExtraction: late fusion sample-count mismatch between voice and EEG");

        FeatureSet fused;
        fused.label = vfs.label + "-fused-late";
        fused.vectors.resize(vfs.vectors.size());
        for (size_t i = 0; i < vfs.vectors.size(); ++i)
        {
            fused.vectors[i] = vfs.vectors[i];
            fused.vectors[i].insert(
                fused.vectors[i].end(), efs.vectors[i].begin(), efs.vectors[i].end());
        }
        result.push_back(std::move(fused));
    }
    return result;
}

} // namespace thesis
