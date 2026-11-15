// VoiceDysphonia.hpp: simulated dysphonia by editing a voice's F0 and loudness (WORLD).
//
// A requested jitter of 3 % is not the jitter of the output: the synthesis, the pulse shape and
// the estimator all move it. These tests therefore MEASURE what the output carries, on a steady
// synthetic vowel whose true cycles are known, and compare with the request: that is the closed
// loop the thesis reports its levels from (the proposal's "measured units").
//
// The estimator finds the glottal pulse instants of a signal from the smoothed |y| (one maximum
// per cycle, refined by a parabola) and derives local jitter, local shimmer and the F0 contour
// from them. It is checked against the CONTROL (no edit), whose jitter it must call small.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "../lib/include/VoiceDysphonia.hpp"

using namespace thesis;

namespace
{

constexpr double kFs = 44100.0;
constexpr double kF0 = 140.0;
constexpr double kSeconds = 1.6;

/// An impulse train at kF0 through three resonances (a steady /a/-like vowel), peak 0.5.
std::vector<float> steady_vowel(double fs, double seconds, double f0)
{
    const auto n = static_cast<std::size_t>(seconds * fs);
    std::vector<double> x(n, 0.0);
    const double period = fs / f0;
    for (std::size_t k = 0; static_cast<double>(k) * period < static_cast<double>(n) - 2.0; ++k)
    {
        const double t = static_cast<double>(k) * period;
        const auto i = static_cast<std::size_t>(t);
        const double frac = t - static_cast<double>(i); // split between two samples: exact period
        x[i] += 1.0 - frac;
        x[i + 1] += frac;
    }
    // {centre Hz, bandwidth Hz} of three resonators.
    const std::array<std::array<double, 2>, 3> formants{
        {{700.0, 80.0}, {1200.0, 90.0}, {2600.0, 120.0}}};
    for (const auto& formant : formants)
    {
        const double r = std::exp(-std::numbers::pi * formant[1] / fs);
        const double a1 = -2.0 * r * std::cos(2.0 * std::numbers::pi * formant[0] / fs);
        const double a2 = r * r;
        double y1 = 0.0;
        double y2 = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double y = x[i] - a1 * y1 - a2 * y2;
            y2 = y1;
            y1 = y;
            x[i] = y;
        }
    }
    double peak = 0.0;
    for (double v : x) peak = std::max(peak, std::fabs(v));
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(0.5 * x[i] / peak);
    return out;
}

struct Pulses
{
    std::vector<double> time_s;    ///< instant of each glottal pulse
    std::vector<double> amplitude; ///< |y| at it
};

/// One pulse per cycle: maxima of |y| smoothed over 0.2 ms, at least `min_gap_fraction` of a
/// nominal period apart (the stronger of two closer maxima wins), refined by a parabola through
/// the maximum and its neighbours.
Pulses find_pulses(
    const std::vector<float>& y, double fs, double f0_nominal, double min_gap_fraction = 0.7)
{
    const auto n = y.size();
    const int half = std::max(1, static_cast<int>(0.0001 * fs));
    std::vector<double> e(n, 0.0);
    {
        std::vector<double> a(n);
        for (std::size_t i = 0; i < n; ++i) a[i] = std::fabs(static_cast<double>(y[i]));
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::size_t lo = i >= static_cast<std::size_t>(half) ? i - half : 0;
            const std::size_t hi = std::min(n - 1, i + half);
            double acc = 0.0;
            for (std::size_t k = lo; k <= hi; ++k) acc += a[k];
            e[i] = acc / static_cast<double>(hi - lo + 1);
        }
    }
    const double peak_e = *std::max_element(e.begin(), e.end());
    const auto min_gap = static_cast<std::size_t>(min_gap_fraction * fs / f0_nominal);
    std::vector<std::size_t> at; // accepted maxima, in samples
    for (std::size_t i = 1; i + 1 < n; ++i)
    {
        if (!(e[i] >= e[i - 1] && e[i] > e[i + 1]) || e[i] < 0.25 * peak_e) continue;
        if (!at.empty() && i - at.back() < min_gap)
        {
            if (e[i] > e[at.back()]) at.back() = i; // the stronger of two close maxima is the pulse
            continue;
        }
        at.push_back(i);
    }
    Pulses p;
    for (const std::size_t i : at)
    {
        const double denom = e[i - 1] - 2.0 * e[i] + e[i + 1];
        const double shift = denom != 0.0 ? 0.5 * (e[i - 1] - e[i + 1]) / denom : 0.0;
        p.time_s.push_back((static_cast<double>(i) + shift) / fs);
        p.amplitude.push_back(std::fabs(static_cast<double>(y[i])));
    }
    return p;
}

/// F0 (Hz) of 40 ms windows hopped by 10 ms inside [from_s, to_s]: the lag, between 70 and 300 Hz,
/// with the highest autocorrelation. Insensitive to how many humps one cycle shows.
std::vector<double> windowed_f0(const std::vector<float>& y, double fs, double from_s, double to_s)
{
    const auto win = static_cast<std::size_t>(0.04 * fs);
    const auto hop = static_cast<std::size_t>(0.01 * fs);
    const auto lag_min = static_cast<std::size_t>(fs / 300.0);
    const auto lag_max = static_cast<std::size_t>(fs / 70.0);
    std::vector<double> f0;
    for (auto start = static_cast<std::size_t>(from_s * fs);
        start + win + lag_max < y.size() && static_cast<double>(start + win) / fs <= to_s;
        start += hop)
    {
        // A periodic signal correlates as well at 2, 3... periods as at one: take the SHORTEST
        // lag that is a local maximum within 10 % of the best (measured: the plain argmax
        // returned the 70 Hz subharmonic for most windows).
        std::vector<double> r_of(lag_max + 1, -1.0);
        double best = -1.0;
        for (std::size_t lag = lag_min; lag <= lag_max; ++lag)
        {
            double num = 0.0;
            double e0 = 0.0;
            double e1 = 0.0;
            for (std::size_t i = start; i < start + win; ++i)
            {
                const double a = y[i];
                const double b = y[i + lag];
                num += a * b;
                e0 += a * a;
                e1 += b * b;
            }
            const double r = num / std::sqrt(e0 * e1 + 1e-30);
            r_of[lag] = r;
            best = std::max(best, r);
        }
        std::size_t chosen = lag_min;
        for (std::size_t lag = lag_min + 1; lag < lag_max; ++lag)
            if (r_of[lag] >= 0.9 * best && r_of[lag] >= r_of[lag - 1] && r_of[lag] >= r_of[lag + 1])
            {
                chosen = lag;
                break;
            }
        f0.push_back(fs / static_cast<double>(chosen));
    }
    return f0;
}

/// Intervals between consecutive pulses inside [from_s, to_s].
std::vector<double> intervals(const Pulses& p, double from_s, double to_s)
{
    std::vector<double> t;
    for (std::size_t i = 0; i + 1 < p.time_s.size(); ++i)
        if (p.time_s[i] >= from_s && p.time_s[i + 1] <= to_s)
            t.push_back(p.time_s[i + 1] - p.time_s[i]);
    return t;
}

double mean(const std::vector<double>& v)
{
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(std::max<std::size_t>(v.size(), 1));
}

/// Local jitter, percent: mean |T[c+1] - T[c]| / mean T.
double local_jitter_pct(const std::vector<double>& t)
{
    std::vector<double> d;
    for (std::size_t i = 0; i + 1 < t.size(); ++i) d.push_back(std::fabs(t[i + 1] - t[i]));
    return 100.0 * mean(d) / mean(t);
}

/// Local shimmer, dB: mean |20 log10(A[c+1]/A[c])| over pulses in [from_s, to_s].
double local_shimmer_db(const Pulses& p, double from_s, double to_s)
{
    std::vector<double> a;
    for (std::size_t i = 0; i < p.time_s.size(); ++i)
        if (p.time_s[i] >= from_s && p.time_s[i] <= to_s) a.push_back(p.amplitude[i]);
    std::vector<double> d;
    for (std::size_t i = 0; i + 1 < a.size(); ++i)
        d.push_back(std::fabs(20.0 * std::log10(a[i + 1] / a[i])));
    return mean(d);
}

double rms(const std::vector<float>& v)
{
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    return std::sqrt(s / static_cast<double>(v.size()));
}

DysphoniaParams control()
{
    return DysphoniaParams{};
}

DysphoniaParams rough(double jitter, double shimmer)
{
    DysphoniaParams p;
    p.profile = DysphoniaProfile::kRough;
    p.jitter_pct = jitter;
    p.shimmer_db = shimmer;
    return p;
}

// The part of the signal the measures use: the first and last 0.2 s carry WORLD's edge effects.
constexpr double kFrom = 0.2;
constexpr double kTo = kSeconds - 0.2;

} // namespace

TEST(VoiceDysphoniaParams, ProfilesAndKindsParseAndAnUnknownOneNamesTheValidOnes)
{
    EXPECT_EQ(parse_dysphonia_profile("rough"), DysphoniaProfile::kRough);
    EXPECT_EQ(parse_dysphonia_profile("voice_breaks"), DysphoniaProfile::kVoiceBreaks);
    EXPECT_EQ(parse_break_kind("aphonic"), BreakKind::kAphonic);
    try
    {
        (void) parse_dysphonia_profile("hoarse");
        FAIL() << "an unknown profile must be refused";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("diplophonia"), std::string::npos) << e.what();
    }
    EXPECT_THROW((void) parse_break_kind("stutter"), std::invalid_argument);
}

TEST(VoiceDysphoniaParams, ValuesOutsideTheLimitsAreRefusedWithTheKeyName)
{
    check_dysphonia_params(control());
    check_dysphonia_params(rough(2.0, 1.0));
    EXPECT_THROW(
        check_dysphonia_params(rough(0.0, 1.0)), std::invalid_argument); // that is the control
    EXPECT_THROW(check_dysphonia_params(rough(25.0, 1.0)), std::invalid_argument);
    DysphoniaParams t;
    t.profile = DysphoniaProfile::kTremor;
    t.tremor_f0_depth_pct = 4.0;
    t.tremor_rate_hz = 1.0; // below the 2-10 Hz of Kreiman and Gerratt's tremors
    EXPECT_THROW(check_dysphonia_params(t), std::invalid_argument);
    DysphoniaParams b;
    b.profile = DysphoniaProfile::kVoiceBreaks;
    b.breaks_per_second = 5.0;
    b.break_ms = 300.0; // 1.5 s of break per second of voice
    EXPECT_THROW(check_dysphonia_params(b), std::invalid_argument);
}

TEST(VoiceDysphoniaRefusals, AnAnalysisRateAtOrBelow8kHzIsRefusedBecauseD4CWouldReturnAWhisper)
{
    const auto x = steady_vowel(8000.0, 0.5, kF0);
    EXPECT_THROW((void) degrade_voice_f0(x, 8000.0, 8000.0, control(), 1u), std::invalid_argument);
    // Stored at 44.1 kHz but recorded at 8 kHz: the analysis rate is the recording's.
    const auto y = steady_vowel(kFs, 0.5, kF0);
    EXPECT_THROW((void) degrade_voice_f0(y, kFs, 8000.0, control(), 1u), std::invalid_argument);
    EXPECT_THROW((void) degrade_voice_f0(y, 16000.0, kFs, control(), 1u), std::invalid_argument);
    EXPECT_THROW((void) degrade_voice_f0({}, kFs, kFs, control(), 1u), std::invalid_argument);
}

TEST(VoiceDysphoniaRefusals, AnUnvoicedUtteranceIsAnErrorNotAPassThrough)
{
    const std::vector<float> silence(static_cast<std::size_t>(kFs * 0.5), 0.0F);
    EXPECT_THROW((void) degrade_voice_f0(silence, kFs, kFs, control(), 1u), std::runtime_error);
}

TEST(VoiceDysphoniaControl, KeepsTheLengthTheLoudnessAndMovesNoJitter)
{
    const auto x = steady_vowel(kFs, kSeconds, kF0);
    const auto y = degrade_voice_f0(x, kFs, kFs, control(), 1u);
    ASSERT_EQ(y.size(), x.size());
    EXPECT_NEAR(20.0 * std::log10(rms(y) / rms(x)), 0.0, 0.5); // RMS is matched by construction

    const auto in = find_pulses(x, kFs, kF0);
    const auto out = find_pulses(y, kFs, kF0);
    const double jitter_in = local_jitter_pct(intervals(in, kFrom, kTo));
    const double jitter_out = local_jitter_pct(intervals(out, kFrom, kTo));
    EXPECT_LT(jitter_in, 0.1) << "the estimator calls a perfectly periodic vowel irregular";
    EXPECT_LT(jitter_out, 0.6) << "the resynthesis alone must not add jitter";
}

TEST(VoiceDysphoniaControl, ARoundTripThroughANativeRateKeepsTheLength)
{
    const auto x = steady_vowel(kFs, 1.0, kF0);
    const auto y = degrade_voice_f0(x, kFs, 16000.0, control(), 1u);
    EXPECT_EQ(y.size(), x.size());
    EXPECT_NEAR(20.0 * std::log10(rms(y) / rms(x)), 0.0, 1.0);
}

TEST(VoiceDysphoniaDeterminism, TheSameSeedGivesTheSameSignalAndAnotherSeedAnotherOne)
{
    const auto x = steady_vowel(kFs, 0.8, kF0);
    const auto a = degrade_voice_f0(x, kFs, kFs, rough(3.0, 1.0), 5u);
    const auto b = degrade_voice_f0(x, kFs, kFs, rough(3.0, 1.0), 5u);
    const auto c = degrade_voice_f0(x, kFs, kFs, rough(3.0, 1.0), 6u);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(VoiceDysphoniaDeterminism, ConcurrentCallsGiveWhatSerialCallsGive)
{
    // WORLD keeps no global state (each call owns its RandnState); this proves it for the
    // pipeline as the run uses it, under OpenMP, with a different seed per call.
    const auto x = steady_vowel(kFs, 0.6, kF0);
    constexpr int kCalls = 8;
    std::vector<std::vector<float>> serial(kCalls);
    std::vector<std::vector<float>> concurrent(kCalls);
    for (int i = 0; i < kCalls; ++i)
        serial[static_cast<std::size_t>(i)] =
            degrade_voice_f0(x, kFs, kFs, rough(3.0, 1.0), static_cast<std::uint32_t>(i + 1));
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < kCalls; ++i)
        concurrent[static_cast<std::size_t>(i)] =
            degrade_voice_f0(x, kFs, kFs, rough(3.0, 1.0), static_cast<std::uint32_t>(i + 1));
    for (int i = 0; i < kCalls; ++i)
        EXPECT_EQ(serial[static_cast<std::size_t>(i)], concurrent[static_cast<std::size_t>(i)])
            << "call " << i << " depends on the thread schedule";
}

TEST(VoiceDysphoniaCalibration, RoughJitterAndShimmerLandNearTheRequest)
{
    // The request is not the output: the proposal reports MEASURED values, so the test pins
    // how far they may sit from it on a steady vowel (measured 2026-11-14: 3 -> 2.6, 6 -> 5.5).
    const auto x = steady_vowel(kFs, kSeconds, kF0);
    double previous = 0.0;
    for (const double requested : {1.0, 3.0, 6.0})
    {
        const auto y = degrade_voice_f0(x, kFs, kFs, rough(requested, 1.5), 11u);
        const auto pulses = find_pulses(y, kFs, kF0);
        const double measured = local_jitter_pct(intervals(pulses, kFrom, kTo));
        EXPECT_NEAR(measured, requested, 0.25 * requested + 0.3) << "requested " << requested;
        EXPECT_GT(measured, previous) << "the levels must stay ordered";
        previous = measured;
        const double shimmer = local_shimmer_db(pulses, kFrom, kTo);
        EXPECT_NEAR(shimmer, 1.5, 0.5) << "requested " << requested << " % jitter, shimmer 1.5 dB";
    }
}

TEST(VoiceDysphoniaCalibration, TremorSwaysTheF0ByTheRequestedDepthAtTheRequestedRate)
{
    const auto x = steady_vowel(kFs, kSeconds, kF0);
    DysphoniaParams p;
    p.profile = DysphoniaProfile::kTremor;
    p.tremor_f0_depth_pct = 8.0;
    p.tremor_rate_hz = 5.0;
    const auto y = degrade_voice_f0(x, kFs, kFs, p, 3u);
    const auto pulses = find_pulses(y, kFs, kF0);

    // Least-squares fit of a 5 Hz sinusoid to the F0 of every cycle (1 / interval).
    double f_mean = 0.0;
    std::vector<std::pair<double, double>> samples; // (time, F0)
    for (std::size_t i = 0; i + 1 < pulses.time_s.size(); ++i)
        if (pulses.time_s[i] >= kFrom && pulses.time_s[i + 1] <= kTo)
        {
            const double t = 0.5 * (pulses.time_s[i] + pulses.time_s[i + 1]);
            samples.emplace_back(t, 1.0 / (pulses.time_s[i + 1] - pulses.time_s[i]));
            f_mean += samples.back().second;
        }
    ASSERT_GT(samples.size(), 100u);
    f_mean /= static_cast<double>(samples.size());
    double sc = 0.0;
    double ss = 0.0;
    double cc = 0.0;
    double sn = 0.0;
    double cn = 0.0;
    for (const auto& [t, f] : samples)
    {
        const double w = 2.0 * std::numbers::pi * 5.0 * t;
        sc += std::sin(w) * std::cos(w);
        ss += std::sin(w) * std::sin(w);
        cc += std::cos(w) * std::cos(w);
        sn += (f - f_mean) * std::sin(w);
        cn += (f - f_mean) * std::cos(w);
    }
    const double det = ss * cc - sc * sc;
    const double a = (sn * cc - cn * sc) / det;
    const double b = (cn * ss - sn * sc) / det;
    const double depth_pct = 100.0 * std::sqrt(a * a + b * b) / f_mean;
    EXPECT_NEAR(depth_pct, 8.0, 2.0);
    EXPECT_NEAR(f_mean, kF0, 0.04 * kF0);
}

TEST(VoiceDysphoniaCalibration, PitchJumpsPlaceTheExpectedNumberOfBreaks)
{
    // 1.6 s of voice at 1.2 breaks per second: ceil(1.92) = 2 breaks of 100 ms, an up and a down.
    const auto x = steady_vowel(kFs, kSeconds, kF0);
    DysphoniaParams p;
    p.profile = DysphoniaProfile::kVoiceBreaks;
    p.break_kind = BreakKind::kPitchJump;
    p.breaks_per_second = 1.2;
    p.break_ms = 100.0;
    const auto y = degrade_voice_f0(x, kFs, kFs, p, 4u);
    // Pulse picking is the wrong ruler here: a x0.65 cycle (11 ms) shows two similar humps of
    // |y|, so a pulse finder reads 7.4 / 3.6 ms alternations (measured) that are not in the
    // signal. The F0 of 40 ms windows (autocorrelation) sees the period, not the humps.
    const auto f0 = windowed_f0(y, kFs, kFrom, kTo);
    // Windows whose F0 is off the nominal by more than 20 %, grouped in runs; the first run
    // must be the up break, the second the down break. A run's value is its MEDIAN F0 / nominal:
    // the window that straddles a break edge reads a mixture of the two regimes (measured: 70,
    // 300 and 232 Hz at the edges) and would drag a mean.
    std::vector<double> run_ratio;
    std::vector<double> run;
    const auto close_run = [&]
    {
        if (run.empty()) return;
        std::sort(run.begin(), run.end());
        run_ratio.push_back(run[run.size() / 2] / kF0);
        run.clear();
    };
    for (const double f : f0)
    {
        if (f < 0.8 * kF0 || f > 1.25 * kF0)
            run.push_back(f);
        else
            close_run();
    }
    close_run();
    ASSERT_EQ(run_ratio.size(), 2u);
    EXPECT_NEAR(run_ratio[0], kPitchJumpUp, 0.15);
    EXPECT_NEAR(run_ratio[1], kPitchJumpDown, 0.1);
}

TEST(VoiceDysphoniaCalibration, DiplophoniaAlternatesTheCyclesByTheRequestedPeriod)
{
    // Alternate cycles +6 % / -6 %: consecutive periods differ by about 12 %.
    const auto x = steady_vowel(kFs, kSeconds, kF0);
    DysphoniaParams p;
    p.profile = DysphoniaProfile::kDiplophonia;
    p.period_alt_pct = 6.0;
    p.amp_alt_db = 3.0;
    const auto y = degrade_voice_f0(x, kFs, kFs, p, 9u);
    const auto pulses = find_pulses(y, kFs, kF0);
    const double jitter = local_jitter_pct(intervals(pulses, kFrom, kTo));
    EXPECT_NEAR(jitter, 12.0, 3.0);
}
