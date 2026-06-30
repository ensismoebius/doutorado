#include "E05FeatureExtraction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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
    using wavelets::WaveletTraits;
    using wavelets::Daub4;
    using wavelets::PACKET_WAVELET;

    auto& coeffs = WaveletTraits<Daub4>::coeffs;
    std::span<const double> filter(coeffs.data(), coeffs.size());

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

    for (const auto& group : groups)
    {
        if (want_energy)  features.push_back(compute_energy(group));
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

// Pull the raw 1-D signal for the requested modality from a sample:
//   "eeg"   → EEG channel
//   "voice" → audio channel
//   "fused" → audio if present, otherwise EEG
// Returns 256 zeros when the chosen modality is absent, so downstream transforms
// always receive a non-empty signal. Shared by both extraction strategies.
std::vector<double> signal_for_modality(const E05Sample& sample, const std::string& modality)
{
    auto present = [](const nn::Tensor& t) { return t.rows() > 0 && t.cols() > 0; };

    std::vector<double> sig;
    if (modality == "eeg")
    {
        if (present(sample.eeg)) sig = tensor_to_vec(sample.eeg);
    }
    else if (modality == "voice")
    {
        if (present(sample.audio)) sig = tensor_to_vec(sample.audio);
    }
    else // "fused"
    {
        if (present(sample.audio))    sig = tensor_to_vec(sample.audio);
        else if (present(sample.eeg)) sig = tensor_to_vec(sample.eeg);
    }

    if (sig.empty()) sig.assign(256, 0.0);
    return sig;
}

// Nominal sample rates by modality (Hz).
double modality_sample_rate(const std::string& modality)
{
    if (modality == "eeg") return 800.0;
    return 22050.0; // "voice" and "fused" use voice rate
}

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

nn::Tensor vec_to_column_tensor(const std::vector<double>& sig)
{
    nn::Tensor t(static_cast<nn::Index>(sig.size()), 1);
    for (nn::Index i = 0; i < static_cast<nn::Index>(sig.size()); ++i)
        t.at(i, 0) = static_cast<float>(sig[static_cast<size_t>(i)]);
    return t;
}
} // namespace

auto extract_features(const E05DatasetView& view,
    const E05Config::FeatureExtraction& cfg,
    const E05Config::Training& training,
    const std::string& modality) -> std::vector<FeatureSet>
{
    std::vector<FeatureSet> result;

    if (cfg.strategy == "handcrafted")
    {
        const double sample_rate = modality_sample_rate(modality);

        FeatureSet fs;
        fs.label = "handcrafted-" + cfg.handcrafted.scale;
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
            std::vector<double> sig = signal_for_modality(sample, modality);

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
            std::vector<double> sig = signal_for_modality(sample, modality);
            max_len = std::max(max_len, sig.size());
            raw_signals.push_back(std::move(sig));
        }

        if (max_len == 0)
            throw std::runtime_error("E05FeatureExtraction: no valid raw signals for autoencoder");

        std::vector<nn::Tensor> train_samples;
        train_samples.reserve(raw_signals.size());
        for (auto& sig : raw_signals)
        {
            sig.resize(max_len, 0.0);
            train_samples.push_back(vec_to_column_tensor(sig));
        }

        const int hidden_size = first_encoder_dim(cfg.autoencoder.encoder_layer_spec, 64);
        const int latent_size = last_encoder_dim(cfg.autoencoder.encoder_layer_spec, 16);
        const int num_layers = std::max<int>(1, static_cast<int>(cfg.autoencoder.encoder_layer_spec.size() / 2));

        nn::models::lstm::LSTMAutoencoderConfig ae_cfg;
        ae_cfg.input_size = 1;
        ae_cfg.seq_len = static_cast<int>(max_len);
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
        fs.label = "autoencoder-lstm";
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

} // namespace e05
