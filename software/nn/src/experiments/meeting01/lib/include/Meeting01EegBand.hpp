#pragma once
// Meeting01EegBand.hpp -- the band every EEG recording is filtered to, and how much a window of
// it can hold.
//
// Why the second half exists. An autoencoder is only tested if its window holds MORE than the
// bottleneck can carry; otherwise a plain linear projection copies the window through whole and
// every family is judged on how close its optimizer gets to zero. A signal that holds nothing
// outside a band of B Hz, observed for T seconds, has about 2 x B x T independent values (the
// time-bandwidth product). For a 256-sample window, band 0.5-40 Hz (B = 39.5):
//
//     rate      window    2 x B x T    latent 64 is ...
//     160 Hz    1.6 s     126.4        a real bottleneck (EEGMMIDB; PCA k=64 test MSE .19)
//     512 Hz    0.5 s      39.5        wider than the signal (Siena until results_format 13:
//                                      PCA k=64 test MSE 0.0000, top 64 PCs held 100% of the
//                                      training variance, measured 2026-11-10 on fold 0)
//
// That failure is silent: nothing throws, the grid runs for weeks, and the Siena column reads
// "PCA 0.000, every network above it". Meeting01Config::validate() therefore refuses an EEG
// dataset whose window has no more degrees of freedom than its latent (check_dataset).

#include <string>

namespace meeting01
{

// Standard EEG analysis band: 0.5-40 Hz is the practice found for Siena and for EEG deep-learning
// preprocessing in general (.wiki/Core/DataLoaders.md, "Transforms").
inline constexpr double kEegBandLowHz = 0.5;
inline constexpr double kEegBandHighHz = 40.0;

// The datasets EegWindowDataset loads (Meeting01Dataset.cpp, load_grouped_windows).
[[nodiscard]] inline auto is_eeg_dataset(const std::string& name) -> bool
{
    return name == "eegmmidb" || name == "siena";
}

// 2 x bandwidth x duration of a window of `window_size` samples at `sample_rate` Hz once it is
// band-limited to [kEegBandLowHz, kEegBandHighHz]: 126.4 at 160 Hz, 39.5 at 512 Hz (256 samples).
// An approximation (the filter's edges are not brick walls), used as a floor the latent must stay
// strictly below, not as an exact rank.
[[nodiscard]] constexpr auto eeg_window_degrees_of_freedom(int window_size, double sample_rate)
    -> double
{
    return 2.0 * (kEegBandHighHz - kEegBandLowHz) * static_cast<double>(window_size) / sample_rate;
}

} // namespace meeting01
