/**
 * @file src/experiments/thesis/lib/include/VoiceDegradation.hpp
 * @brief Simulated dysphonia: the voice whispered by a controllable fraction.
 *
 * Why. The thesis is about speakers whose voice fails, but the dataset holds healthy voices
 * only. A dysphonic voice keeps the shape of the vocal tract (the formants: which vowel,
 * whose mouth) and loses the regular vibration of the vocal folds — it turns breathy, and in
 * the extreme (aphonia) it is a whisper. That is what this models.
 *
 * How, for one signal x (44.1 kHz voice):
 *
 *   x ──frames of 25 ms, Hann, 50 % overlap──> per frame: LPC envelope 1/A(z), energy E
 *   white noise ──through 1/A(z)──> same envelope, no pitch ──scaled to E──> overlap-add = w
 *   output = sqrt(1 - level) · x  +  sqrt(level) · w
 *
 * x and w are uncorrelated, so the output keeps the power of x for every level, and `level`
 * is the fraction of that power that is breath instead of voice:
 *
 *   level   0.25        0.5        0.75       1.0
 *   voice   75 %        50 %       25 %       0 %  (a whisper: aphonia)
 *
 * The LPC order is 2 + fs/1000 (46 at 44.1 kHz), the usual rule of thumb for an all-pole
 * vocal-tract model (Rabiner & Schafer, 2011). A frame of zero energy stays silent.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace thesis
{

/// The whispered twin of `x` (level = 1 of the mix): same frame energies and spectral
/// envelopes, white-noise excitation. Deterministic in `seed`.
auto whisper(const std::vector<float>& x, double sample_rate_hz, std::uint32_t seed)
    -> std::vector<float>;

/// sqrt(1 - level) * x + sqrt(level) * whisper(x). 0 < level <= 1, else std::invalid_argument.
auto degrade_voice(
    const std::vector<float>& x, double sample_rate_hz, float level, std::uint32_t seed)
    -> std::vector<float>;

} // namespace thesis
