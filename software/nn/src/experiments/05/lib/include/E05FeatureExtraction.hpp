#pragma once

#include <string>
#include <vector>

#include "E05Config.hpp"
#include "E05Dataset.hpp"

namespace e05
{

// A named feature set: the strategy+scale label and the extracted vectors.
struct FeatureSet
{
    std::string label;                       // e.g. "handcrafted-lfcc-energy_zcr"
    std::vector<std::vector<double>> vectors; // one per sample, aligned with E05DatasetView::samples
};

// Extract handcrafted features (DTWPT + descriptors) from a single signal.
// signal:      raw samples as doubles.
// cfg:         handcrafted extraction config.
// sample_rate: Hz of the signal (22050 for voice, 800 for EEG).
//              Used to map DTWPT sub-bands to Bark/MEL/LFCC frequency groups.
// Returns a flat feature vector.
auto extract_handcrafted(const std::vector<double>& signal,
    const E05Config::HandcraftedConfig& cfg,
    double sample_rate) -> std::vector<double>;

// Extract features from all samples using the configured strategy.
// modality: "voice" | "eeg" | "fused" — determines which tensor is processed
//           and the sample rate passed to extract_handcrafted.
// Returns one FeatureSet per strategy evaluated.
auto extract_features(const E05DatasetView& view,
    const E05Config::FeatureExtraction& cfg,
    const std::string& modality) -> std::vector<FeatureSet>;

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

} // namespace e05
