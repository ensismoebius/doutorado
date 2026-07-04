#include "E05FeatureExtraction.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "progress/ProgressManager.hpp"
#include "wavelet/Types.hpp"
#include "wavelet/WaveletTransformResults.hpp"
#include "wavelet/waveletOperations.hpp"

namespace e05
{

// ─── scalar descriptors ────────────────────────────────────────────────────

auto compute_zcr(const std::vector<double>& signal) -> double
{
    if (signal.size() < 2) return 0.0;
    int crossings = 0;
    for (size_t i = 1; i < signal.size(); ++i)
    {
        if ((signal[i] >= 0.0) != (signal[i - 1] >= 0.0))
            ++crossings;
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
        if (k > 0)
            diff_sum += std::abs(signal[peaks[k]] - signal[peaks[k - 1]]);
    }
    double mean_A = amp_sum / static_cast<double>(peaks.size());
    if (mean_A == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return diff_sum / (static_cast<double>(peaks.size() - 1) * mean_A);
}

void apply_preemphasis(std::vector<double>& signal, double alpha)
{
    for (size_t n = signal.size(); n-- > 1;)
        signal[n] -= alpha * signal[n - 1];
}

// ─── frequency scale helpers ────────────────────────────────────────────────

namespace
{

// Zwicker & Terhardt (1980) Bark approximation.
double hz_to_bark(double f)
{
    return 13.0 * std::atan(0.00076 * f) +
           3.5  * std::atan(std::pow(f / 7500.0, 2.0));
}

// O'Shaughnessy MEL formula.
double hz_to_mel(double f)
{
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

// Runtime name → mother-wavelet decomposition filter. Coefficient arrays have
// static storage (constexpr in Types.hpp), so returning spans over them is safe.
// Only the tags listed here (all with WaveletTraits specializations) are valid;
// E05Config::validate() rejects any other name before extraction runs.
std::span<const double> wavelet_filter(const std::string& name)
{
    using namespace wavelets;
    static const std::unordered_map<std::string, std::span<const double>> table = {
        {"haar",   get_wavelet<Haar>()},
        {"daub4",  get_wavelet<Daub4>()},   {"daub6",  get_wavelet<Daub6>()},
        {"daub8",  get_wavelet<Daub8>()},   {"daub10", get_wavelet<Daub10>()},
        {"daub12", get_wavelet<Daub12>()},  {"daub14", get_wavelet<Daub14>()},
        {"daub16", get_wavelet<Daub16>()},  {"daub18", get_wavelet<Daub18>()},
        {"daub20", get_wavelet<Daub20>()},  {"daub22", get_wavelet<Daub22>()},
        {"daub24", get_wavelet<Daub24>()},  {"daub26", get_wavelet<Daub26>()},
        {"daub28", get_wavelet<Daub28>()},  {"daub30", get_wavelet<Daub30>()},
        {"daub32", get_wavelet<Daub32>()},  {"daub34", get_wavelet<Daub34>()},
        {"daub36", get_wavelet<Daub36>()},  {"daub38", get_wavelet<Daub38>()},
        {"daub40", get_wavelet<Daub40>()},  {"daub42", get_wavelet<Daub42>()},
        {"daub44", get_wavelet<Daub44>()},  {"daub46", get_wavelet<Daub46>()},
    };
    const auto it = table.find(name);
    if (it == table.end())
        throw std::invalid_argument("E05FeatureExtraction: unknown wavelet \"" + name + "\"");
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
            acc += x[n] * std::cos(M_PI / static_cast<double>(N) *
                                   (static_cast<double>(n) + 0.5) * static_cast<double>(k));
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
std::vector<std::vector<double>> group_by_scale(
    wavelets::WaveletTransformResults& result,
    long n_parts,
    const std::string& scale, double sample_rate)
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

    const int n_bands    = (scale == "bark") ? 24 : 20;
    const double nyquist = sample_rate / 2.0;
    const double max_sv  = (scale == "bark") ? hz_to_bark(nyquist) : hz_to_mel(nyquist);

    std::vector<std::vector<double>> groups(static_cast<size_t>(n_bands));

    for (long p = 0; p < n_parts; ++p)
    {
        // Center frequency of sub-band p (uniform partition of Nyquist).
        double center_hz = (p + 0.5) * nyquist / static_cast<double>(n_parts);
        double sv        = (scale == "bark") ? hz_to_bark(center_hz) : hz_to_mel(center_hz);
        int band         = static_cast<int>(sv / max_sv * n_bands);
        band             = std::clamp(band, 0, n_bands - 1);

        auto coefs = wavelets::WaveletTransformResults::get_wavelet_packet_transforms(
            result.transformedSignal, p, result.levelsOfTransformation);
        auto& g = groups[static_cast<size_t>(band)];
        g.insert(g.end(), coefs.begin(), coefs.end());
    }

    // Drop empty bins — may occur when n_parts < n_bands (low DTWPT levels).
    groups.erase(
        std::remove_if(groups.begin(), groups.end(),
            [](const std::vector<double>& g) { return g.empty(); }),
        groups.end());

    return groups;
}

} // namespace

// ─── handcrafted extraction ─────────────────────────────────────────────────

auto extract_handcrafted(const std::vector<double>& signal,
    const E05Config::HandcraftedConfig& cfg,
    double sample_rate) -> std::vector<double>
{
    using wavelets::PACKET_WAVELET;

    const std::span<const double> filter = wavelet_filter(cfg.wavelet);

    auto result = wavelets::malat(signal, filter, PACKET_WAVELET,
        static_cast<unsigned int>(cfg.dtwpt_level));

    long n_parts = result.get_wavelet_packet_amount_of_parts();

    // Group sub-bands according to the perceptual frequency scale.
    auto groups = group_by_scale(result, n_parts, cfg.scale, sample_rate);

    const auto& descs = cfg.descriptors;
    bool want_energy  = std::find(descs.begin(), descs.end(), "energy")  != descs.end();
    bool want_zcr     = std::find(descs.begin(), descs.end(), "zcr")     != descs.end();
    bool want_entropy = std::find(descs.begin(), descs.end(), "entropy") != descs.end();
    bool want_teager  = std::find(descs.begin(), descs.end(), "teager")  != descs.end();
    bool want_jitter  = std::find(descs.begin(), descs.end(), "jitter")  != descs.end();
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
        if (want_zcr)     features.push_back(compute_zcr(group));
        if (want_entropy) features.push_back(compute_entropy(group));
        if (want_teager)  features.push_back(compute_teager(group));
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
        for (long j = 0; j < t.cols(); ++j)
            v.push_back(static_cast<double>(t.at(i, j)));
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
using SignalGetter = std::function<std::vector<double>(const E05Sample&)>;

std::vector<double> voice_signal(const E05Sample& sample)
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

std::vector<double> eeg_signal(const E05Sample& sample)
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
std::vector<double> fused_early_signal(const E05Sample& sample)
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
    try { return std::stoi(token); } catch (...) { return fallback; }
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
                static_cast<float>(sig[static_cast<size_t>(f) * static_cast<size_t>(frame_len)
                                       + static_cast<size_t>(w)]);
    return t;
}

// Cap on the AE sequence length: signals are windowed into at most this many
// frames, so seq_len stays small regardless of raw signal length.
constexpr int kAeMaxFrames = 64;
} // namespace

namespace
{

// Runs the configured strategy (handcrafted or autoencoder) against whatever
// single signal get_signal() returns per sample. Used directly for "voice"
// and "eeg" modality, and reused twice by late fusion (once per signal) and
// once by early fusion (on the pre-concatenated signal). label_suffix tags
// the resulting FeatureSet so voice-part/EEG-part/fused variants stay
// distinguishable in results output.
auto extract_features_core(const E05DatasetView& view,
    const E05Config::FeatureExtraction& cfg,
    const E05Config::Training& training,
    const SignalGetter& get_signal,
    double sample_rate,
    const std::string& label_suffix) -> std::vector<FeatureSet>
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
        nn::progress::ProgressManager::instance().set_description(
            feat_bar, "DTWPT | scale=" + cfg.handcrafted.scale +
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
        if (cfg.autoencoder.model != "lstm-ae")
        {
            throw std::runtime_error(
                "E05FeatureExtraction: only lstm-ae is implemented for autoencoder feature learning");
        }

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
            throw std::runtime_error("E05FeatureExtraction: no valid raw signals for autoencoder");

        // Window each signal into at most kAeMaxFrames frames of frame_len samples.
        // input_size = frame_len (features per step), seq_len = T_frames. This bounds
        // the LSTM unroll length regardless of raw signal size, and gives the AE a
        // real per-step feature vector instead of a single scalar.
        const int frame_len = std::max<int>(1,
            static_cast<int>((max_len + kAeMaxFrames - 1) / kAeMaxFrames));
        const int T_frames = kAeMaxFrames;
        const size_t padded = static_cast<size_t>(T_frames) * static_cast<size_t>(frame_len);

        std::vector<nn::Tensor> train_samples;
        train_samples.reserve(raw_signals.size());
        for (auto& sig : raw_signals)
        {
            sig.resize(padded, 0.0);
            train_samples.push_back(vec_to_frame_tensor(sig, T_frames, frame_len));
        }

        const int hidden_size = first_encoder_dim(cfg.autoencoder.encoder_layer_spec, 64);
        const int latent_size = last_encoder_dim(cfg.autoencoder.encoder_layer_spec, 16);
        const int num_layers = std::max<int>(1, static_cast<int>(cfg.autoencoder.encoder_layer_spec.size() / 2));

        nn::models::lstm::LSTMAutoencoderConfig ae_cfg;
        ae_cfg.input_size = frame_len;
        ae_cfg.seq_len = T_frames;
        ae_cfg.hidden_size = hidden_size;
        ae_cfg.latent_size = latent_size;
        ae_cfg.num_layers = num_layers;

        nn::models::lstm::LSTMAutoencoder model(ae_cfg);

        nn::training::TrainerConfig trainer_cfg;
        trainer_cfg.epochs = training.epochs;
        trainer_cfg.learning_rate = training.learning_rate;
        trainer_cfg.batch_size = training.samples_per_batch;

        nn::training::Trainer<nn::models::lstm::LSTMAutoencoder> trainer(model, trainer_cfg);
        (void) trainer.fit_autoencoder(train_samples);

        FeatureSet fs;
        fs.label = "autoencoder-lstm" + label_suffix;
        fs.vectors.reserve(train_samples.size());
        for (const auto& sample : train_samples)
        {
            auto latent = model.encode(sample, false);
            fs.vectors.push_back(tensor_to_vec(latent));
        }

        result.push_back(std::move(fs));
    }
    else
    {
        throw std::invalid_argument(
            "E05FeatureExtraction: unknown strategy \"" + cfg.strategy + "\"");
    }

    return result;
}

} // namespace

auto extract_features(const E05DatasetView& view,
    const E05Config::FeatureExtraction& cfg,
    const E05Config::Training& training,
    const std::string& modality,
    const std::string& fusion_mode) -> std::vector<FeatureSet>
{
    if (modality == "voice")
        return extract_features_core(view, cfg, training, voice_signal, kVoiceSampleRate, "");

    if (modality == "eeg")
        return extract_features_core(view, cfg, training, eeg_signal, kEegSampleRate, "");

    if (modality != "fused")
        throw std::invalid_argument("E05FeatureExtraction: unknown modality \"" + modality + "\"");

    if (fusion_mode == "early")
    {
        return extract_features_core(
            view, cfg, training, fused_early_signal, kVoiceSampleRate, "-fused-early");
    }

    if (fusion_mode != "late")
        throw std::invalid_argument("E05FeatureExtraction: unknown fusion_mode \"" + fusion_mode + "\"");

    // Late fusion: extract independently per signal, then concatenate the
    // resulting feature vectors sample-by-sample (audit C12).
    auto voice_sets = extract_features_core(view, cfg, training, voice_signal, kVoiceSampleRate, "");
    auto eeg_sets = extract_features_core(view, cfg, training, eeg_signal, kEegSampleRate, "");

    if (voice_sets.size() != eeg_sets.size())
        throw std::runtime_error(
            "E05FeatureExtraction: late fusion produced mismatched FeatureSet counts");

    std::vector<FeatureSet> result;
    result.reserve(voice_sets.size());
    for (size_t k = 0; k < voice_sets.size(); ++k)
    {
        auto& vfs = voice_sets[k];
        auto& efs = eeg_sets[k];
        if (vfs.vectors.size() != efs.vectors.size())
            throw std::runtime_error(
                "E05FeatureExtraction: late fusion sample-count mismatch between voice and EEG");

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

} // namespace e05
