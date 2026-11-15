/**
 * @file src/experiments/thesis/lib/include/DegradeVoices.hpp
 * @brief Replace the voice of every sample of a view by its simulated-dysphonia version.
 *
 * One function, shared by the thesis binary and thesis_features (the identity test reads the
 * same degraded voices the experiment does, not a re-implementation of them).
 */
#pragma once

#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"

namespace thesis
{

/// The voice of every sample replaced by its dysphonic version (degradation.profile), in place.
/// whisper_mix: VoiceDegradation.hpp; every other profile: VoiceDysphonia.hpp (WORLD). One random
/// stream per sample (seed mixed with its index), so the result does not depend on the thread
/// schedule. An exception cannot leave an OpenMP region, so the first one is kept and rethrown
/// after the loop, with the sample it came from: a voice that cannot carry the requested
/// dysphonia (no voiced frame) stops the run instead of passing through unchanged.
void degrade_voices(ThesisDatasetView& view, const ThesisConfig::Degradation& dg);

} // namespace thesis
