#include "E05FeatureExtraction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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
    // Detect positive zero-crossings as proxy for period boundaries.
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

// ─── handcrafted extraction ─────────────────────────────────────────────────

auto extract_handcrafted(const std::vector<double>& signal,
    const E05Config::HandcraftedConfig& cfg) -> std::vector<double>
{
    using wavelets::WaveletTraits;
    using wavelets::Daub4;
    using wavelets::PACKET_WAVELET;

    // Use Daubechies-4 for DTWPT (good time-frequency localisation).
    auto& coeffs = WaveletTraits<Daub4>::coeffs;
    std::span<const double> filter(coeffs.data(), coeffs.size());

    auto result = wavelets::malat(signal, filter, PACKET_WAVELET,
        static_cast<unsigned int>(cfg.dtwpt_level));

    long n_parts = result.get_wavelet_packet_amount_of_parts();
    long sig_len = static_cast<long>(signal.size());

    std::vector<double> features;

    const auto& descs = cfg.descriptors;
    bool want_energy  = std::find(descs.begin(), descs.end(), "energy")  != descs.end();
    bool want_zcr     = std::find(descs.begin(), descs.end(), "zcr")     != descs.end();
    bool want_entropy = std::find(descs.begin(), descs.end(), "entropy") != descs.end();
    bool want_teager  = std::find(descs.begin(), descs.end(), "teager")  != descs.end();
    bool want_jitter  = std::find(descs.begin(), descs.end(), "jitter")  != descs.end();
    bool want_shimmer = std::find(descs.begin(), descs.end(), "shimmer") != descs.end();

    for (long p = 0; p < n_parts; ++p)
    {
        long start = p * (sig_len / n_parts);
        long end   = (p + 1) * (sig_len / n_parts);
        auto subband = result.get_wavelet_packet_transforms(start, end, sig_len);

        if (want_energy)  features.push_back(compute_energy(subband));
        if (want_zcr)     features.push_back(compute_zcr(subband));
        if (want_entropy) features.push_back(compute_entropy(subband));
        if (want_teager)  features.push_back(compute_teager(subband));
        if (want_jitter)
        {
            double j = compute_jitter(subband, 44100.0);
            features.push_back(std::isnan(j) ? 0.0 : j);
        }
        if (want_shimmer)
        {
            double s = compute_shimmer(subband, 44100.0);
            features.push_back(std::isnan(s) ? 0.0 : s);
        }
    }

    return features;
}

// ─── batch extraction ───────────────────────────────────────────────────────

namespace
{
// Convert nn::Tensor (float) to std::vector<double>.
std::vector<double> tensor_to_vec(const nn::Tensor& t)
{
    std::vector<double> v;
    v.reserve(static_cast<size_t>(t.size()));
    for (long i = 0; i < t.rows(); ++i)
        for (long j = 0; j < t.cols(); ++j)
            v.push_back(static_cast<double>(t.at(i, j)));
    return v;
}
} // namespace

auto extract_features(const E05DatasetView& view,
    const E05Config::FeatureExtraction& cfg) -> std::vector<FeatureSet>
{
    std::vector<FeatureSet> result;

    if (cfg.strategy == "handcrafted")
    {
        FeatureSet fs;
        fs.label = "handcrafted-" + cfg.handcrafted.scale;
        fs.vectors.reserve(view.samples.size());

        const auto n_samples = static_cast<long>(view.samples.size());
        uint32_t feat_bar = nn::progress::ProgressManager::instance().create_bar(
            "Feature extraction", static_cast<float>(n_samples));
        nn::progress::ProgressManager::instance().set_description(
            feat_bar, "DTWPT | scale=" + cfg.handcrafted.scale +
                      "  descriptors=" + std::to_string(cfg.handcrafted.descriptors.size()));

        // Pre-size so parallel index assignment is safe (no push_back races).
        fs.vectors.resize(static_cast<size_t>(n_samples));
        long feat_done = 0;

        #pragma omp parallel for schedule(dynamic, 4) shared(feat_done)
        for (long i = 0; i < n_samples; ++i)
        {
            const auto& sample = view.samples[static_cast<size_t>(i)];
            std::vector<double> sig;
            if (sample.audio.rows() > 0 && sample.audio.cols() > 0)
                sig = tensor_to_vec(sample.audio);
            else if (sample.eeg.rows() > 0 && sample.eeg.cols() > 0)
                sig = tensor_to_vec(sample.eeg);
            else
                sig.assign(256, 0.0);

            size_t n = 1;
            while (n < sig.size()) n <<= 1;
            sig.resize(n, 0.0);

            fs.vectors[static_cast<size_t>(i)] = extract_handcrafted(sig, cfg.handcrafted);

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
        // Autoencoder feature extraction is handled by the training loop via Trainer.
        // Here we return an empty set as placeholder; callers invoke train_autoencoder()
        // and then supply the latent vectors back as a FeatureSet.
        FeatureSet fs;
        fs.label = "autoencoder-" + cfg.autoencoder.model;
        // Vectors populated externally after autoencoder training.
        result.push_back(std::move(fs));
    }
    else
    {
        throw std::invalid_argument("E05FeatureExtraction: unknown strategy " + cfg.strategy);
    }

    return result;
}

} // namespace e05
