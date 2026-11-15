#include "VoiceDysphonia.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>

#include "utility/Resample.hpp"
#include "world/cheaptrick.h"
#include "world/d4c.h"
#include "world/harvest.h"
#include "world/synthesis.h"

namespace thesis
{

namespace
{

constexpr double kAnalysisFramePeriodMs = 5.0;   ///< WORLD's default analysis grid
constexpr double kControlFramePeriodMs = 1.0;    ///< the grid the F0 is edited on
constexpr int kFramesPerAnalysisFrame = 5;       ///< 5 ms / 1 ms
constexpr double kF0FloorHz = 71.0;              ///< WORLD's default floor: covers a 106 Hz voice
constexpr double kTremorAmplitudePhaseRad = 0.8; ///< amplitude sway is not in phase with F0's
// Local jitter is the mean absolute FIRST DIFFERENCE of the periods. For periods 1 + sigma N_c with
// iid N_c ~ N(0, 1) the difference is N(0, 2 sigma^2), whose mean absolute value is
// sigma * 2 / sqrt(pi): the factor that turns a requested local jitter (or shimmer, the same
// statistic of the per-cycle dB values) into the sigma of the draws.
constexpr double kMeanAbsDifferenceOfGaussians = 1.1283791670955126;

using Matrix = std::vector<std::vector<double>>;

auto row_pointers(Matrix& m) -> std::vector<double*>
{
    std::vector<double*> p(m.size());
    for (std::size_t i = 0; i < m.size(); ++i) p[i] = m[i].data();
    return p;
}

auto rms_of(const std::vector<double>& v, std::size_t n) -> double
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += v[i] * v[i];
    return std::sqrt(s / static_cast<double>(std::max<std::size_t>(n, 1)));
}

/// What WORLD found in the voice, on the 1 ms grid the edits are made on.
struct Analysis
{
    int n = 0;  ///< samples of the input
    int fs = 0; ///< analysis rate, Hz
    int fft_size = 0;
    int frames = 0;         ///< 1 ms frames
    std::vector<double> f0; ///< per 1 ms frame; 0 = unvoiced
    Matrix spectrum;        ///< (frames, fft/2+1)
    Matrix aperiodicity;    ///< (frames, fft/2+1)
};

auto analyse(const std::vector<double>& x, int fs) -> Analysis
{
    const int n = static_cast<int>(x.size());

    HarvestOption harvest;
    InitializeHarvestOption(&harvest);
    harvest.frame_period = kAnalysisFramePeriodMs;
    harvest.f0_floor = kF0FloorHz;
    const int coarse = GetSamplesForHarvest(fs, n, kAnalysisFramePeriodMs);
    std::vector<double> t(static_cast<std::size_t>(coarse)), f0(static_cast<std::size_t>(coarse));
    Harvest(x.data(), n, fs, &harvest, t.data(), f0.data());

    if (std::none_of(f0.begin(), f0.end(), [](double v) { return v > 0.0; }))
        throw std::runtime_error(
            "VoiceDysphonia: Harvest found no voiced frame in an utterance of " +
            std::to_string(static_cast<double>(n) / fs) +
            " s, so there is no pitch to edit. The utterance is silent or not voiced; remove it "
            "from the base or choose another degradation profile");

    CheapTrickOption cheap;
    InitializeCheapTrickOption(fs, &cheap);
    cheap.f0_floor = kF0FloorHz;
    cheap.fft_size = GetFFTSizeForCheapTrick(fs, &cheap);
    const int bins = cheap.fft_size / 2 + 1;

    Matrix sp(
        static_cast<std::size_t>(coarse), std::vector<double>(static_cast<std::size_t>(bins)));
    Matrix ap(
        static_cast<std::size_t>(coarse), std::vector<double>(static_cast<std::size_t>(bins)));
    {
        auto sp_rows = row_pointers(sp);
        auto ap_rows = row_pointers(ap);
        CheapTrick(x.data(), n, fs, t.data(), f0.data(), coarse, &cheap, sp_rows.data());
        D4COption d4c;
        InitializeD4COption(&d4c);
        D4C(x.data(), n, fs, t.data(), f0.data(), coarse, cheap.fft_size, &d4c, ap_rows.data());
    }

    // Resample the analysis to 1 ms frames: F0 linearly between two voiced frames (the nearer
    // frame's value next to an unvoiced one), envelope and aperiodicity linearly.
    Analysis a;
    a.n = n;
    a.fs = fs;
    a.fft_size = cheap.fft_size;
    a.frames = (coarse - 1) * kFramesPerAnalysisFrame + 1;
    a.f0.assign(static_cast<std::size_t>(a.frames), 0.0);
    a.spectrum.assign(
        static_cast<std::size_t>(a.frames), std::vector<double>(static_cast<std::size_t>(bins)));
    a.aperiodicity.assign(
        static_cast<std::size_t>(a.frames), std::vector<double>(static_cast<std::size_t>(bins)));
    for (int j = 0; j < a.frames; ++j)
    {
        const int i = j / kFramesPerAnalysisFrame;
        const int r = j % kFramesPerAnalysisFrame;
        const double w = static_cast<double>(r) / kFramesPerAnalysisFrame;
        const int i2 = std::min(i + 1, coarse - 1);
        const double fa = f0[static_cast<std::size_t>(i)];
        const double fb = f0[static_cast<std::size_t>(i2)];
        a.f0[static_cast<std::size_t>(j)] =
            (fa > 0.0 && fb > 0.0) ? (1.0 - w) * fa + w * fb : (w < 0.5 ? fa : fb);
        for (int k = 0; k < bins; ++k)
        {
            const auto ki = static_cast<std::size_t>(k);
            a.spectrum[static_cast<std::size_t>(j)][ki] =
                (1.0 - w) * sp[static_cast<std::size_t>(i)][ki] +
                w * sp[static_cast<std::size_t>(i2)][ki];
            a.aperiodicity[static_cast<std::size_t>(j)][ki] =
                (1.0 - w) * ap[static_cast<std::size_t>(i)][ki] +
                w * ap[static_cast<std::size_t>(i2)][ki];
        }
    }
    return a;
}

/// Breaks: `count` windows of `break_ms`, one in each of `count` equal slots of the voiced
/// span, centred on the voiced frame nearest the slot's middle. Returns the number placed.
auto apply_breaks(std::vector<double>& f0, const DysphoniaParams& p) -> int
{
    const int frames = static_cast<int>(f0.size());
    int first = -1;
    int last = -1;
    int voiced = 0;
    for (int j = 0; j < frames; ++j)
        if (f0[static_cast<std::size_t>(j)] > 0.0)
        {
            if (first < 0) first = j;
            last = j;
            ++voiced;
        }
    const double voiced_s = voiced * kControlFramePeriodMs / 1000.0;
    // Rounded UP: a request for 0.5 breaks per second on a 0.8 s utterance is one break, not
    // zero, because zero would leave the utterance identical to the control without a word.
    const int count =
        std::max(1, static_cast<int>(std::ceil(p.breaks_per_second * voiced_s - 1e-9)));
    const double slot = static_cast<double>(last - first + 1) / count;
    const int half =
        std::max(1, static_cast<int>(std::lround(p.break_ms / 2.0 / kControlFramePeriodMs)));

    const std::vector<double> clean = f0;
    int placed = 0;
    for (int i = 0; i < count; ++i)
    {
        const int mid = first + static_cast<int>((static_cast<double>(i) + 0.5) * slot);
        const int lo = std::max(first, static_cast<int>(first + i * slot));
        const int hi = std::min(last, static_cast<int>(first + (i + 1) * slot) - 1);
        const auto is_voiced = [&clean](int frame)
        { return clean[static_cast<std::size_t>(frame)] > 0.0; };
        int centre = -1;
        for (int d = 0; d <= hi - lo && centre < 0; ++d)
        {
            if (mid - d >= lo && is_voiced(mid - d))
                centre = mid - d;
            else if (mid + d <= hi && is_voiced(mid + d))
                centre = mid + d;
        }
        if (centre < 0) continue; // a slot without a voiced frame carries no break
        ++placed;
        const double factor = (i % 2 == 0) ? kPitchJumpUp : kPitchJumpDown;
        for (int j = std::max(0, centre - half); j < std::min(frames, centre + half); ++j)
        {
            auto& f = f0[static_cast<std::size_t>(j)];
            if (p.break_kind == BreakKind::kAphonic)
                f = 0.0;
            else if (f > 0.0)
                f *= factor;
        }
    }
    return placed;
}

/// Rebuilds the voice from the (edited) analysis and applies the loudness edits. Returns
/// exactly `a.n` samples.
auto resynthesize(const Analysis& a,
    const std::vector<double>& original,
    const DysphoniaParams& p,
    std::uint32_t seed) -> std::vector<double>
{
    const int fs = a.fs;
    std::vector<double> f = a.f0;

    // Pitch edits that act on the contour.
    if (p.profile == DysphoniaProfile::kVoiceBreaks)
    {
        if (apply_breaks(f, p) == 0)
            throw std::runtime_error(
                "VoiceDysphonia: the break rate placed no break: no slot of the voiced span has "
                "a voiced frame. Raise voice_breaks breaks_per_second or use a longer utterance");
    }
    if (p.profile == DysphoniaProfile::kTremor)
    {
        const double depth = p.tremor_f0_depth_pct / 100.0;
        for (std::size_t j = 0; j < f.size(); ++j)
        {
            const double s = static_cast<double>(j) * kControlFramePeriodMs / 1000.0;
            if (f[j] > 0.0)
                f[j] *= 1.0 + depth * std::sin(2.0 * std::numbers::pi * p.tremor_rate_hz * s);
        }
    }

    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Per-cycle period edits (rough, diplophonia). The target is the PULSE INSTANTS: cycle k
    // lasts base_period * factor_k, so the F0 of each 1 ms frame is set to the mean rate that
    // takes the phase from one frame edge to the next along the piecewise-linear phase through
    // those instants. (Scaling the F0 of frames labelled by the UNEDITED cycle grid instead
    // smears each alternation over two cycles: a diplophonia of +-6 % came out at a third of
    // the requested difference.) Pulses are therefore where the request puts them, up to the
    // 1 ms frame; the instants walk away from the unedited ones by sqrt(cycles) * sigma * T,
    // a few ms over a 4 s utterance.
    const bool perturbs_periods =
        p.profile == DysphoniaProfile::kRough || p.profile == DysphoniaProfile::kDiplophonia;
    if (perturbs_periods)
    {
        const double sigma = p.jitter_pct / 100.0 / kMeanAbsDifferenceOfGaussians;
        const double dt = kControlFramePeriodMs / 1000.0;
        const std::vector<double> base = f;
        std::vector<double> edited(f.size(), 0.0);
        long cycle = 0;
        std::size_t j = 0;
        while (j < f.size())
        {
            if (!(base[j] > 0.0))
            {
                ++j;
                continue;
            }
            const std::size_t first = j;
            while (j < f.size() && base[j] > 0.0) ++j;
            const std::size_t last = j - 1; // the voiced run is [first, last]
            const double t_begin = (static_cast<double>(first) - 0.5) * dt;
            const double t_end = (static_cast<double>(last) + 0.5) * dt;
            const auto base_f0_at = [&](double t)
            {
                const double pos =
                    std::clamp(t / dt, static_cast<double>(first), static_cast<double>(last));
                const auto i0 = static_cast<std::size_t>(pos);
                const std::size_t i1 = std::min(i0 + 1, last);
                const double w = pos - static_cast<double>(i0);
                return (1.0 - w) * base[i0] + w * base[i1];
            };
            std::vector<double> pulse{t_begin};
            while (pulse.back() < t_end)
            {
                double factor = 1.0 + sigma * normal(rng);
                if (p.profile == DysphoniaProfile::kDiplophonia)
                    factor *=
                        1.0 + ((cycle % 2 != 0) ? p.period_alt_pct : -p.period_alt_pct) / 100.0;
                ++cycle;
                pulse.push_back(pulse.back() + std::max(factor, 0.5) / base_f0_at(pulse.back()));
            }
            // Cycles completed by time t along the piecewise-linear phase through the pulses.
            const auto phase_at = [&](double t)
            {
                const auto it = std::upper_bound(pulse.begin(), pulse.end(), t);
                const std::size_t k =
                    static_cast<std::size_t>(std::max<std::ptrdiff_t>(it - pulse.begin() - 1, 0));
                return static_cast<double>(k) + (t - pulse[k]) / (pulse[k + 1] - pulse[k]);
            };
            for (std::size_t m = first; m <= last; ++m)
            {
                const double lo = (static_cast<double>(m) - 0.5) * dt;
                const double hi = (static_cast<double>(m) + 0.5) * dt;
                edited[m] = (phase_at(hi) - phase_at(lo)) / dt;
            }
        }
        f = std::move(edited);
    }

    // WORLD synthesis on the 1 ms grid.
    const int frames = static_cast<int>(f.size());
    const int length = static_cast<int>((frames - 1.0) * kControlFramePeriodMs / 1000.0 * fs + 1.0);
    std::vector<double> y(static_cast<std::size_t>(length) + 4096, 0.0);
    {
        // Synthesis takes mutable row pointers though it only reads them.
        Matrix sp = a.spectrum;
        Matrix ap = a.aperiodicity;
        auto sp_rows = row_pointers(sp);
        auto ap_rows = row_pointers(ap);
        Synthesis(f.data(),
            frames,
            sp_rows.data(),
            ap_rows.data(),
            a.fft_size,
            kControlFramePeriodMs,
            fs,
            length,
            y.data());
    }
    y.resize(static_cast<std::size_t>(length));
    const auto L = static_cast<std::size_t>(std::min(length, a.n));

    // Per-sample F0 and phase of the edited contour, then loudness per cycle.
    const bool perturbs_loudness =
        p.profile == DysphoniaProfile::kRough || p.profile == DysphoniaProfile::kDiplophonia;
    std::vector<double> gain_db(L, 0.0);
    std::vector<double> f_sample(L, 0.0);
    std::vector<double> phase_sample(L, 0.0);
    double pp = 0.0;
    for (std::size_t s = 0; s < L; ++s)
    {
        const double pos = static_cast<double>(s) / (fs * kControlFramePeriodMs / 1000.0);
        const auto i = static_cast<std::size_t>(pos);
        const double w = pos - static_cast<double>(i);
        const std::size_t i2 = std::min(i + 1, f.size() - 1);
        const double fa = f[std::min(i, f.size() - 1)];
        const double fb = f[i2];
        f_sample[s] = (fa > 0.0 && fb > 0.0) ? (1.0 - w) * fa + w * fb : (w < 0.5 ? fa : fb);
        if (f_sample[s] > 0.0) pp += 2.0 * std::numbers::pi * f_sample[s] / fs;
        phase_sample[s] = pp;
    }
    if (perturbs_loudness && (p.shimmer_db > 0.0 || p.amp_alt_db > 0.0))
    {
        const int n_cycles = static_cast<int>(pp / (2.0 * std::numbers::pi)) + 3;
        const double sigma_db = p.shimmer_db / kMeanAbsDifferenceOfGaussians;
        std::vector<double> g(static_cast<std::size_t>(n_cycles), 0.0);
        for (int c = 0; c < n_cycles; ++c)
            g[static_cast<std::size_t>(c)] =
                sigma_db * normal(rng) + ((c % 2 != 0) ? 0.5 : -0.5) * p.amp_alt_db;

        // The gain steps sit halfway between the real pulse peaks of the synthesized voice.
        std::vector<int> peak(static_cast<std::size_t>(n_cycles), -1);
        std::vector<double> peak_value(static_cast<std::size_t>(n_cycles), -1.0);
        for (std::size_t s = 0; s < L; ++s)
            if (f_sample[s] > 0.0)
            {
                const auto c = static_cast<std::size_t>(std::min(
                    n_cycles - 1, static_cast<int>(phase_sample[s] / (2.0 * std::numbers::pi))));
                if (std::fabs(y[s]) > peak_value[c])
                {
                    peak_value[c] = std::fabs(y[s]);
                    peak[c] = static_cast<int>(s);
                }
            }
        std::vector<int> peak_at;
        std::vector<int> peak_cycle;
        for (int c = 0; c < n_cycles; ++c)
            if (peak[static_cast<std::size_t>(c)] >= 0)
            {
                peak_at.push_back(peak[static_cast<std::size_t>(c)]);
                peak_cycle.push_back(c);
            }
        for (std::size_t q = 0; q < peak_at.size(); ++q)
        {
            const int begin = q ? (peak_at[q - 1] + peak_at[q]) / 2 : 0;
            const int end =
                (q + 1 < peak_at.size()) ? (peak_at[q] + peak_at[q + 1]) / 2 : static_cast<int>(L);
            for (int s = begin; s < end; ++s)
                gain_db[static_cast<std::size_t>(s)] = g[static_cast<std::size_t>(peak_cycle[q])];
        }
        // 1 ms smoothing of the steps, so a gain change does not click.
        const int half = std::max(1, static_cast<int>(0.0005 * fs));
        std::vector<double> smooth(L, 0.0);
        for (std::size_t s = 0; s < L; ++s)
        {
            double acc = 0.0;
            int cnt = 0;
            const auto lo =
                static_cast<std::size_t>(std::max<long>(0, static_cast<long>(s) - half));
            const auto hi = std::min(L - 1, s + static_cast<std::size_t>(half));
            for (std::size_t k = lo; k <= hi; ++k)
            {
                acc += gain_db[k];
                ++cnt;
            }
            smooth[s] = acc / cnt;
        }
        gain_db = std::move(smooth);
    }
    const bool sways = p.profile == DysphoniaProfile::kTremor;
    const double am_depth = sways ? kTremorAmplitudeToF0Ratio * p.tremor_f0_depth_pct / 100.0 : 0.0;
    for (std::size_t s = 0; s < L; ++s)
    {
        const double am = 1.0 + am_depth * std::sin(2.0 * std::numbers::pi * p.tremor_rate_hz *
                                                        static_cast<double>(s) / fs +
                                                    kTremorAmplitudePhaseRad);
        y[s] *= std::pow(10.0, (f_sample[s] > 0.0 ? gain_db[s] : 0.0) / 20.0) * am;
    }

    // Same loudness as the input (RMS), whatever the edits did to it.
    const double target = rms_of(original, L);
    const double actual = rms_of(y, L);
    if (!(actual > 0.0) || !std::isfinite(actual))
        throw std::runtime_error(
            "VoiceDysphonia: the resynthesized voice is silent or not finite (RMS " +
            std::to_string(actual) + "); the analysis of this utterance failed");
    const double scale = target / actual;

    // The output must be as long as the input. WORLD's grid can end up to one analysis frame
    // short of it; more than that is a failure, not rounding.
    const auto max_gap =
        static_cast<std::size_t>(std::ceil(kAnalysisFramePeriodMs / 1000.0 * fs)) + 1;
    if (static_cast<std::size_t>(a.n) > L + max_gap)
        throw std::runtime_error("VoiceDysphonia: the resynthesis has " + std::to_string(L) +
                                 " samples for an input of " + std::to_string(a.n) +
                                 "; a gap of more than one 5 ms analysis frame is a failure");
    std::vector<double> out(static_cast<std::size_t>(a.n), 0.0);
    for (std::size_t s = 0; s < L; ++s) out[s] = y[s] * scale;
    return out;
}

} // namespace

auto parse_dysphonia_profile(const std::string& name) -> DysphoniaProfile
{
    if (name == "resynthesis_control") return DysphoniaProfile::kResynthesisControl;
    if (name == "rough") return DysphoniaProfile::kRough;
    if (name == "tremor") return DysphoniaProfile::kTremor;
    if (name == "voice_breaks") return DysphoniaProfile::kVoiceBreaks;
    if (name == "diplophonia") return DysphoniaProfile::kDiplophonia;
    throw std::invalid_argument(
        "VoiceDysphonia: unknown profile \"" + name +
        "\"; the F0 profiles are resynthesis_control, rough, tremor, voice_breaks and diplophonia");
}

auto parse_break_kind(const std::string& name) -> BreakKind
{
    if (name == "pitch_jump") return BreakKind::kPitchJump;
    if (name == "aphonic") return BreakKind::kAphonic;
    throw std::invalid_argument(
        "VoiceDysphonia: unknown break kind \"" + name + "\"; use pitch_jump or aphonic");
}

auto dysphonia_params(const ThesisConfig::Degradation& dg) -> DysphoniaParams
{
    const std::string& name = *dg.profile;
    if (name == "whisper_mix")
        throw std::invalid_argument(
            "VoiceDysphonia: degradation.profile \"whisper_mix\" is VoiceDegradation's, not an "
            "F0 profile");
    DysphoniaParams p;
    try
    {
        p.profile = parse_dysphonia_profile(name);
    }
    catch (const std::invalid_argument&)
    {
        throw std::invalid_argument(
            "ThesisConfig: degradation.profile \"" + name +
            "\" is unknown; use whisper_mix, resynthesis_control, rough, tremor, voice_breaks "
            "or diplophonia");
    }
    switch (p.profile)
    {
        case DysphoniaProfile::kResynthesisControl:
            break;
        case DysphoniaProfile::kRough:
            p.jitter_pct = dg.jitter_pct;
            p.shimmer_db = dg.shimmer_db;
            break;
        case DysphoniaProfile::kTremor:
            p.tremor_f0_depth_pct = dg.f0_depth_pct;
            p.tremor_rate_hz = dg.rate_hz;
            break;
        case DysphoniaProfile::kVoiceBreaks:
            p.break_kind = parse_break_kind(*dg.break_kind);
            p.breaks_per_second = dg.breaks_per_second;
            p.break_ms = dg.break_ms;
            break;
        case DysphoniaProfile::kDiplophonia:
            p.period_alt_pct = dg.period_alt_pct;
            p.amp_alt_db = dg.amp_alt_db;
            break;
    }
    return p;
}

namespace
{
void need_in(double v, double lo, double hi, bool lo_open, const char* key)
{
    const bool ok = std::isfinite(v) && (lo_open ? v > lo : v >= lo) && v <= hi;
    if (!ok)
        throw std::invalid_argument(std::string("VoiceDysphonia: ") + key + " must be in " +
                                    (lo_open ? "(" : "[") + std::to_string(lo) + ", " +
                                    std::to_string(hi) + "], got " + std::to_string(v));
}
} // namespace

void check_dysphonia_params(const DysphoniaParams& p)
{
    switch (p.profile)
    {
        case DysphoniaProfile::kResynthesisControl:
            return;
        case DysphoniaProfile::kRough:
            need_in(p.jitter_pct, 0.0, 20.0, true, "jitter_pct");
            need_in(p.shimmer_db, 0.0, 6.0, true, "shimmer_db");
            return;
        case DysphoniaProfile::kTremor:
            need_in(p.tremor_f0_depth_pct, 0.0, 30.0, true, "f0_depth_pct");
            need_in(p.tremor_rate_hz, 2.0, 10.0, false, "rate_hz");
            return;
        case DysphoniaProfile::kVoiceBreaks:
            need_in(p.breaks_per_second, 0.0, 5.0, true, "breaks_per_second");
            need_in(p.break_ms, 20.0, 500.0, false, "break_ms");
            // The windows must fit in the voiced span with room to spare.
            if (p.breaks_per_second * p.break_ms / 1000.0 > 0.5)
                throw std::invalid_argument(
                    "VoiceDysphonia: breaks_per_second * break_ms / 1000 is " +
                    std::to_string(p.breaks_per_second * p.break_ms / 1000.0) +
                    ": more than half of the voice would be a break; lower one of them");
            return;
        case DysphoniaProfile::kDiplophonia:
            need_in(p.period_alt_pct, 0.0, 25.0, true, "period_alt_pct");
            need_in(p.amp_alt_db, 0.0, 12.0, false, "amp_alt_db");
            return;
    }
    throw std::invalid_argument("VoiceDysphonia: unhandled profile");
}

auto degrade_voice_f0(const std::vector<float>& x,
    double input_rate_hz,
    double native_rate_hz,
    const DysphoniaParams& p,
    std::uint32_t seed) -> std::vector<float>
{
    if (x.empty()) throw std::invalid_argument("VoiceDysphonia: the signal is empty");
    if (!(input_rate_hz > 0.0) || !(native_rate_hz > 0.0))
        throw std::invalid_argument("VoiceDysphonia: the sample rates must be positive");
    if (native_rate_hz > input_rate_hz)
        throw std::invalid_argument(
            "VoiceDysphonia: the native rate (" + std::to_string(native_rate_hz) +
            " Hz) is above the stored rate (" + std::to_string(input_rate_hz) + " Hz)");
    if (native_rate_hz <= kMinAnalysisRateHz)
        throw std::invalid_argument(
            "VoiceDysphonia: WORLD's D4C has no aperiodicity band at " +
            std::to_string(native_rate_hz) +
            " Hz (it needs more than 8 kHz): it would return a flat aperiodicity of 1, i.e. a "
            "whisper, not the voice. Use a recording sampled above 8 kHz");
    check_dysphonia_params(p);

    const bool resample = native_rate_hz != input_rate_hz;
    const std::vector<float> at_native =
        resample ? nn::utility::resample_bandlimited(x, input_rate_hz, native_rate_hz) : x;
    std::vector<double> xd(at_native.begin(), at_native.end());
    const int fs = static_cast<int>(std::lround(native_rate_hz));

    const Analysis a = analyse(xd, fs);
    const std::vector<double> yd = resynthesize(a, xd, p, seed);

    std::vector<float> y(yd.begin(), yd.end());
    if (resample)
    {
        y = nn::utility::resample_bandlimited(y, native_rate_hz, input_rate_hz);
        // The round trip can differ by a sample or two; a larger gap is a failure.
        const std::size_t tolerance =
            static_cast<std::size_t>(std::ceil(input_rate_hz / native_rate_hz)) + 2;
        if (y.size() + tolerance < x.size() || y.size() > x.size() + tolerance)
            throw std::runtime_error(
                "VoiceDysphonia: the resample round trip changed the length from " +
                std::to_string(x.size()) + " to " + std::to_string(y.size()));
        y.resize(x.size(), 0.0F);
    }
    return y;
}

} // namespace thesis
