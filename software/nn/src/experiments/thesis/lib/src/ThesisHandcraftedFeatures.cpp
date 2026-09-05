#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ThesisFeatureExtraction.hpp"
#include "wavelet/Types.hpp"
#include "wavelet/WaveletTransformResults.hpp"
#include "wavelet/waveletOperations.hpp"

namespace thesis
{

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

} // namespace thesis
