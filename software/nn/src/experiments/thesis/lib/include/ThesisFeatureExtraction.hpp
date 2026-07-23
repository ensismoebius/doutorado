#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"

namespace thesis
{

// A named feature set: the strategy+scale label and the extracted vectors.
struct FeatureSet
{
    std::string label; // e.g. "handcrafted-lfcc-energy_zcr"
    std::vector<std::vector<double>>
        vectors; // one per sample, aligned with ThesisDatasetView::samples
};

// Extract handcrafted features (DTWPT + descriptors) from a single signal.
// signal:      raw samples as doubles.
// cfg:         handcrafted extraction config.
// sample_rate: Hz of the signal (44100 for voice, 1024 for EEG).
//              Used to map DTWPT sub-bands to Bark/MEL/LFCC frequency groups.
// Returns a flat feature vector.
auto extract_handcrafted(const std::vector<double>& signal,
    const ThesisConfig::HandcraftedConfig& cfg,
    double sample_rate) -> std::vector<double>;

// Extract features from all samples using the configured strategy.
// modality:     "voice" | "eeg" | "fused" — determines which signal(s) feed extraction.
// fusion_mode:  only consulted when modality == "fused":
//                 "early" → voice+EEG raw signals concatenated, extracted once.
//                 "late"  → voice and EEG extracted independently, feature
//                           vectors concatenated afterward.
//               Ignored for modality != "fused".
// Returns one FeatureSet per strategy evaluated.
auto extract_features(const ThesisDatasetView& view,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const std::string& modality,
    const std::string& fusion_mode = "late",
    std::uint32_t seed = 42u) -> std::vector<FeatureSet>;

// Fatal check for the one failure mode a spike loss can hit SILENTLY: if EVERY
// training batch produced an all-zero gradient, the autoencoder trained on nothing and
// its features are meaningless. Throws std::runtime_error naming the cause and the
// fixes; returns normally otherwise (including when only some batches were zero, which
// is legitimate near convergence). Exposed for testing the policy directly.
void assert_gradients_were_live(long backward_calls,
    long zero_grad_calls,
    const std::string& loss_token,
    const std::string& encoding,
    float firing_rate_reg_lambda);

// Helper: compute ZCR for a signal.
auto compute_zcr(const std::vector<double>& signal) -> double;

// Helper: compute Shannon entropy of a signal (normalised to [0,1] prob).
auto compute_entropy(const std::vector<double>& signal) -> double;

// Helper: compute Teager-Kaiser energy operator mean.
auto compute_teager(const std::vector<double>& signal) -> double;

// Helper: compute sub-band energy.
auto compute_energy(const std::vector<double>& subband) -> double;

// Jitter and shimmer require voiced frames; return NaN if fewer than 3 peaks found.
auto compute_jitter(const std::vector<double>& signal, double sample_rate) -> double;
auto compute_shimmer(const std::vector<double>& signal, double sample_rate) -> double;

// In-place first-order pre-emphasis y[n] = x[n] - alpha*x[n-1] (y[0] unchanged).
// Applied to audio signals only, before feature extraction. Iterates back-to-front
// so each x[n-1] is the still-unmodified original sample.
void apply_preemphasis(std::vector<double>& signal, double alpha);

} // namespace thesis
