/**
 * @file src/experiments/thesis/lib/include/VoiceDysphonia.hpp
 * @brief Simulated dysphonia by editing a voice's pitch and loudness, cycle by cycle (WORLD).
 *
 * THE PROBLEM. VoiceDegradation.hpp turns a voice into a whisper: it can only make it breathy
 * or aphonic. A voice also fails by WOBBLING: the vocal folds open and close a little unevenly
 * (rough), the pitch drifts slowly (tremor), the pitch jumps or the voice drops out (breaks),
 * or every second cycle differs from its neighbour (diplophonia). None of these is a noise
 * added to the signal; each is a change in WHEN each glottal cycle happens and HOW LOUD it is.
 *
 * THE METHOD. WORLD (Morise et al. 2016) takes a voice apart into three things and rebuilds it:
 *
 *   voice ──Harvest──> F0 contour (pitch, one value per 5 ms)
 *         ──CheapTrick─> spectral envelope (which vowel, whose mouth)
 *         ──D4C────────> aperiodicity (how noisy each band is)
 *   (F0 edited, envelope and aperiodicity kept) ──Synthesis──> a voice with the same mouth
 *                                                              and a different larynx
 *
 * The 5 ms grid is too coarse for per-cycle effects (a 140 Hz voice has a cycle every 7 ms),
 * so the analysis is resampled to 1 ms frames and the F0 contour is perturbed cycle by cycle
 * on that grid. Loudness (shimmer, amplitude alternation, tremor) is applied afterwards to the
 * synthesized signal, with the gain steps placed between its real pulse peaks.
 *
 *   cycle:        1     2     3     4          jitter   : each cycle's LENGTH is random
 *   F0 edited :  |--|---|-|----|--|---|        shimmer  : each cycle's LOUDNESS is random
 *   loudness  :  ▆     ▄     ▇     ▅          tremor   : F0 (and loudness) sway at ~5 Hz
 *
 * THE PROFILES (parameters are the MEASURED quantities, calibrated by a closed loop in the
 * tests, thesis_voice_dysphonia_gtest; the requested value is not the realised one):
 *
 *   resynthesis_control  analysis and resynthesis, nothing edited (the arm every other
 *                        profile is compared with: the dysphonia effect is the difference
 *                        between a profile and this control, never between a profile and the
 *                        raw voice, because resynthesis alone changes the signal)
 *   rough                jitter_pct (local jitter) and shimmer_db (local shimmer)
 *   tremor               F0 depth and rate; the amplitude depth is not a free knob: it is
 *                        kTremorAmplitudeToF0Ratio times the F0 depth (Kreiman & Gerratt 2005
 *                        note that much "amplitude tremor" is an artefact of F0 modulation)
 *   voice_breaks         breaks_per_second breaks of break_ms, either pitch jumps (F0 x1.5
 *                        then x0.65, alternating) or aphonic gaps (F0 = 0: no voicing)
 *   diplophonia          period_alt_pct and amp_alt_db: every second cycle longer/softer
 *
 * ERRORS, NOT FALLBACKS. Every situation in which the result would silently not be the
 * requested dysphonia throws (std::invalid_argument for a bad request, std::runtime_error for
 * a signal that cannot carry it): an analysis rate of 8 kHz or less (D4C then has zero bands
 * and returns aperiodicity 1 everywhere, i.e. a whisper), an utterance without a voiced frame,
 * a break rate that places no break, an output whose length is not the input's.
 *
 * Deterministic in `seed`, free of global state (every call owns its buffers and random
 * streams), so concurrent calls under OpenMP do not interact.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ThesisConfig.hpp"

namespace thesis
{

enum class DysphoniaProfile
{
    kResynthesisControl,
    kRough,
    kTremor,
    kVoiceBreaks,
    kDiplophonia,
};

enum class BreakKind
{
    kPitchJump,
    kAphonic,
};

/// Fixed (not configurable) constants of the profiles; documented here so that a profile that
/// declares the same keys always means the same signal.
inline constexpr double kTremorAmplitudeToF0Ratio = 2.5; ///< amplitude depth / F0 depth
inline constexpr double kPitchJumpUp = 1.5;              ///< F0 factor of an odd break
inline constexpr double kPitchJumpDown = 0.65;           ///< F0 factor of an even break
/// WORLD's D4C has no band at or below this rate (its band count is
/// floor(min(15000, fs/2 - 3000) / 3000)), so the aperiodicity is a flat 1: a whisper.
inline constexpr double kMinAnalysisRateHz = 8000.0;

/// The values of one profile. Fields a profile does not read stay 0 and are never read.
struct DysphoniaParams
{
    DysphoniaProfile profile = DysphoniaProfile::kResynthesisControl;

    double jitter_pct = 0.0; ///< rough: mean |T[c+1]-T[c]| / mean T of the output, percent
    double shimmer_db = 0.0; ///< rough: mean |20 log10(A[c+1]/A[c])|, dB

    double tremor_f0_depth_pct = 0.0; ///< tremor: peak F0 deviation, percent of F0
    double tremor_rate_hz = 0.0;      ///< tremor: modulation rate

    BreakKind break_kind = BreakKind::kPitchJump; ///< voice_breaks
    double breaks_per_second = 0.0;               ///< voice_breaks: per second of voiced speech
    double break_ms = 0.0;                        ///< voice_breaks: length of one break

    double period_alt_pct = 0.0; ///< diplophonia: odd cycles +x %, even cycles -x % of the period
    double amp_alt_db = 0.0;     ///< diplophonia: odd cycles +x/2 dB, even cycles -x/2 dB
};

/// "resynthesis_control" | "rough" | "tremor" | "voice_breaks" | "diplophonia".
/// Anything else throws std::invalid_argument naming the valid values.
auto parse_dysphonia_profile(const std::string& name) -> DysphoniaProfile;

/// "pitch_jump" | "aphonic"; anything else throws std::invalid_argument.
auto parse_break_kind(const std::string& name) -> BreakKind;

/// The values of a `degradation` block whose profile is one of the F0 profiles, reading only
/// the keys that profile declares. Throws std::invalid_argument for "whisper_mix" (another
/// generator, VoiceDegradation.hpp) and for an unknown name, naming all six valid profiles.
auto dysphonia_params(const ThesisConfig::Degradation& dg) -> DysphoniaParams;

/// Checks the values the chosen profile reads; throws std::invalid_argument naming the key.
/// ThesisConfig::validate calls it on the values a profile declares, so the limits live here.
void check_dysphonia_params(const DysphoniaParams& p);

/// The voice `x`, recorded at `input_rate_hz`, with the dysphonia of `p`.
///
/// WORLD analyses it at `native_rate_hz`, the rate the recording really carries: a signal
/// upsampled from 16 kHz to 44.1 kHz has an empty 8-22 kHz band, and D4C would estimate its
/// aperiodicity bands there from nothing. When the two rates differ, `x` is band-limited
/// resampled down, processed and resampled back; the output has the length of `x`.
///
/// Throws std::invalid_argument for an invalid request (see check_dysphonia_params, a rate
/// that is not positive, `native_rate_hz` above `input_rate_hz`, native rate <= 8 kHz) and
/// std::runtime_error when the signal cannot carry the request (no voiced frame, no break
/// placed, silent output).
auto degrade_voice_f0(const std::vector<float>& x,
    double input_rate_hz,
    double native_rate_hz,
    const DysphoniaParams& p,
    std::uint32_t seed) -> std::vector<float>;

} // namespace thesis
