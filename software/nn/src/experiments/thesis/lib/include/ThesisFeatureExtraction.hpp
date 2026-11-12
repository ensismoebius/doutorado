#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"
#include "tensor/Tensor.hpp"

namespace thesis
{

// A named feature set: the strategy+scale label and the extracted vectors.
struct FeatureSet
{
    std::string label; // e.g. "handcrafted-lfcc-energy_zcr"
    std::vector<std::vector<double>>
        vectors; // one per sample, aligned with ThesisDatasetView::samples
};

// Fired once per trained autoencoder INSIDE extract_features (autoencoder strategy
// only), right after training finishes, with that model's full parameter map
// (Module::state_dict()) before it goes out of scope. `part_tag` distinguishes which
// signal the model was trained on: "" for a single model (voice-only, eeg-only,
// early-fused), "voice"/"eeg" for late fusion's two independently-trained models.
// Exists so a caller can capture weights the AE path would otherwise discard —
// extract_features itself never persists anything (.wiki/Experiments/ParaconsistentGA-Design.md
// §6.2 "why weights aren't saved by default").
using ModelSnapshotFn =
    std::function<void(const std::string& part_tag, const std::map<std::string, nn::Tensor>&)>;

// Extract handcrafted features (DTWPT + descriptors) from a single signal.
// signal:      raw samples as doubles.
// cfg:         handcrafted extraction config.
// sample_rate: Hz of the signal (44100 for voice, 1024 for EEG).
//              Used to map DTWPT sub-bands to Bark/MEL/LFCC frequency groups.
// Returns a flat feature vector.
auto extract_handcrafted(const std::vector<double>& signal,
    const ThesisConfig::HandcraftedConfig& cfg,
    double sample_rate) -> std::vector<double>;

// The input the autoencoders and PCA read for ONE signal: its Welch log power spectrum
// (frames of 512 samples, hop 256, symmetric Hann window, bins 1..256, log10), min-max
// normalized to [0, 1]: 256 values. Why a spectrum and not the waveform is told where it
// is computed (log_power_spectrum, ThesisFeatureExtractionInternal.hpp). Public so that
// thesis_spectrum_parity_gtest can hold it against scipy.signal.welch
// (scripts/testing/gen_thesis_spectrum_refs.py). Refuses a signal shorter than 512
// samples, a non-finite sample, a bin of zero power and a flat spectrum.
auto spectral_input(const std::vector<double>& signal) -> std::vector<float>;

// Extract features from all samples with cfg.feature_extraction's strategy
// (handcrafted | autoencoder | pca). Reads, from `cfg`, only what that strategy uses:
//   dataset.modality   "voice" | "eeg" | "fused" — which signal(s) feed extraction;
//   dataset.fusion_mode (fused only) "early" → voice+EEG raw signals concatenated,
//                       extracted once; "late" → each signal extracted on its own,
//                       feature vectors concatenated afterward;
//   training.*, experiment.seed   autoencoder strategy only (the AE trains with them).
// A config built in code therefore needs exactly those fields declared.
// Returns one FeatureSet per strategy evaluated.
// on_model_trained: fired once per trained model (see ModelSnapshotFn above); pass an
// empty function when no caller wants the weights.
auto extract_features(
    const ThesisDatasetView& view, const ThesisConfig& cfg, const ModelSnapshotFn& on_model_trained)
    -> std::vector<FeatureSet>;

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
