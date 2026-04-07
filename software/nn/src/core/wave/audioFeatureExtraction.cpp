/**
 * @file audioFeatureExtraction.cpp
 * @brief Audio feature extraction routines (framing, windowing, FFT-based features, etc.).
 */

#include "nn/wave/audioFeatureExtraction.h"

#include <fftw3.h> // For FFTW library functions

#include <algorithm> // For std::min, std::max
#include <cmath>     // For cosf, roundf, floorf, logf, sqrtf
#include <cstddef>   // For size_t
#include <numbers>   // For std::numbers::pi_v, std::numbers::sqrt2_v
#include <vector>    // For std::vector

#include "nn/tensor/Tensor.hpp" // For Tensor
#include "nn/wave/filter_operations.hpp"

using std::size_t;
using std::vector;

namespace nn::core::wave
{

/**
 * @brief Step 1: pre-emphasis.
 *
 * Applies pre-emphasis to boost high-frequency components.
 * Formula: y[n] = x[n] - alpha * x[n-1].
 * This compensates for natural high-frequency roll-off in speech spectra.
 *
 * @param signal Audio signal (modified in place).
 * @param coefficient Pre-emphasis coefficient.
 */
void pre_emphasis_inplace(vector<float>& signal, float coefficient)
{
    if (signal.empty())
    {
        return; // Nothing to do for an empty signal
    }
    // Apply pre-emphasis filter in reverse order to avoid temporary copies.
    for (size_t i = signal.size() - 1; i > 0; --i)
    {
        signal[i] = signal[i] - (coefficient * signal[i - 1]);
    }
    // signal[0] remains unchanged.
}

/**
 * @brief Step 2: framing and windowing.
 *
 * Splits the signal into short frames and applies a Hamming window to reduce
 * spectral leakage. The signal is assumed quasi-stationary within each frame.
 *
 * @param signal Audio signal.
 * @param context Context containing framing parameters and outputs.
 * @return Windowed signal frames.
 */
auto framing_and_window(const vector<float>& signal, FramingConfig& context)
    -> vector<vector<float>>
{
    if (signal.empty())
    {
        return {}; // Return empty frames for an empty signal
    }

    if (context.loading_params.constants.ms_to_seconds_factor <= 0.0F)
    {
        throw std::invalid_argument("ms_to_seconds_factor must be > 0");
    }
    if (context.loading_params.audio_params.target_sampling_rate <= 0)
    {
        throw std::invalid_argument("target_sampling_rate must be > 0");
    }
    if (!std::isfinite(context.loading_params.audio_params.frame_duration_ms) ||
        context.loading_params.audio_params.frame_duration_ms <= 0.0F)
    {
        throw std::invalid_argument("frame_duration_ms must be finite and > 0");
    }
    if (!std::isfinite(context.loading_params.audio_params.frame_shift_ms) ||
        context.loading_params.audio_params.frame_shift_ms <= 0.0F)
    {
        throw std::invalid_argument("frame_shift_ms must be finite and > 0");
    }

    // Frame length in samples (output parameter).
    context.frame_length = static_cast<int>(
        roundf(context.loading_params.audio_params.frame_duration_ms *
               static_cast<float>(context.loading_params.audio_params.target_sampling_rate) /
               context.loading_params.constants.ms_to_seconds_factor));

    // Step size between consecutive frames in samples (output parameter).
    context.frame_step = static_cast<int>(
        roundf(context.loading_params.audio_params.frame_shift_ms *
               static_cast<float>(context.loading_params.audio_params.target_sampling_rate) /
               context.loading_params.constants.ms_to_seconds_factor));

    if (context.frame_length <= 0 || context.frame_step <= 0)
    {
        throw std::invalid_argument("Computed frame_length and frame_step must be > 0");
    }

    // Input signal length.
    const int signal_length = static_cast<int>(signal.size());

    // Total number of frames extracted from the signal.
    const int number_of_frames =
        1 + std::max(0, (signal_length - context.frame_length) / context.frame_step);

    // Padded length required to hold all frames.
    const int padded_length = (number_of_frames * context.frame_step) + context.frame_length;

    // Copy signal with zero-padding at the end.
    vector<float> padded_signal(padded_length, 0.0F);
    std::copy(signal.begin(), signal.end(), padded_signal.begin());

    // Hamming window coefficients.
    vector<float> window_function(context.frame_length);

    // Output frame matrix.
    vector<vector<float>> frames(number_of_frames, vector<float>(context.frame_length));

    // Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
    for (int i = 0; i < context.frame_length; ++i)
    {
        window_function[i] = context.loading_params.hamming_window_config.alpha -
                             (context.loading_params.hamming_window_config.beta *
                                 cosf(2 * std::numbers::pi_v<float> * static_cast<float>(i) /
                                      (static_cast<float>(context.frame_length) - 1))); // Hamming
    }

    // Apply window to each frame.
    for (int i = 0; i < number_of_frames; ++i)
    {
        // Start index for current frame.
        const int start_index = i * context.frame_step;

        // Fill current frame with windowed samples.
        for (int j = 0; j < context.frame_length; ++j)
        {
            frames[i][j] = padded_signal[start_index + j] * window_function[j];
        }
    }
    return frames;
}

/**
 * @brief Steps 3 and 4: STFT and power spectrum.
 *
 * Computes the Short-Time Fourier Transform (STFT) for each frame and then
 * derives the power spectrum.
 * STFT is computed with FFTW.
 * Power spectrum is |X[k]|^2, where X[k] is the STFT output.
 *
 * @param frames Windowed signal frames.
 * @param fft_points Number of FFT points (typically frame length).
 * @return Power spectrum for each frame.
 */
auto rfft_power(const vector<vector<float>>& frames, int fft_points) -> nn::Tensor
{
    if (frames.empty())
    {
        throw std::invalid_argument("Input frames cannot be empty.");
    }

    // Number of input frames.
    const size_t number_of_frames = frames.size();

    // Number of frequency bins from FFT (N/2 + 1).
    const size_t number_of_bins = (fft_points / 2) + 1;

    // Output matrix storing per-frame power spectra.
    nn::Tensor power_spectrum(static_cast<int>(number_of_frames), static_cast<int>(number_of_bins));

    // FFTW input buffer (real-valued).
    auto* fftw_input = static_cast<float*>(fftwf_malloc(sizeof(float) * fft_points));

    // FFTW output buffer (complex-valued).
    auto* fftw_output =
        static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * (fft_points / 2 + 1)));

    // FFTW real-to-complex plan.
    fftwf_plan fftw_plan =
        fftwf_plan_dft_r2c_1d(fft_points, fftw_input, fftw_output, FFTW_ESTIMATE);

    // Process each frame.
    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Current frame length.
        const size_t frame_length = frames[frame_index].size();

        // FFT size as `size_t` for safe comparisons.
        const auto fft_points_size_t = static_cast<size_t>(fft_points);

        // Number of samples copied to FFT input (min(frame_length, fft_points)).
        const size_t copy_length = std::min(frame_length, fft_points_size_t);

        // Copy current frame into FFT input buffer.
        std::copy_n(frames[frame_index].begin(), copy_length, fftw_input);

        // Zero-pad when frame is shorter than FFT size.
        std::fill(fftw_input + copy_length, fftw_input + fft_points_size_t, 0.0F);

        // Execute FFT.
        fftwf_execute(fftw_plan);

        // Compute power spectrum for current frame.
        for (size_t bin_index = 0; bin_index < number_of_bins; ++bin_index)
        {
            // Real part for current frequency bin.
            const float real_part = fftw_output[bin_index][0];

            // Imaginary part for current frequency bin.
            const float imaginary_part = fftw_output[bin_index][1];

            // Normalized power spectrum value.
            power_spectrum(static_cast<long>(frame_index), static_cast<long>(bin_index)) =
                (real_part * real_part + imaginary_part * imaginary_part) /
                static_cast<float>(fft_points);
        }
    }

    // Release FFTW resources.
    fftwf_destroy_plan(fftw_plan);
    fftwf_free(fftw_input);
    fftwf_free(fftw_output);

    // Return computed power spectrum.
    return power_spectrum;
}

/**
 * @brief Build the linear triangular filterbank.
 *
 * Unlike MFCC (Mel scale), LFCC uses linearly spaced filters in frequency.
 *
 * @param fft_points Number of FFT points.
 * @param context Context with parameters and output buffers.
 */
void build_linear_filterbank(int fft_points, FilterbankConfig& context)
{
    // Number of FFT bins (N/2 + 1).
    // A bin represents a narrow frequency interval in FFT output.
    const int number_of_bins = (fft_points / 2) + 1;

    // Bin indices corresponding to center frequencies.
    vector<int> bin_indices(context.loading_params.audio_params.number_of_filters + 2);

    // Maximum represented frequency (Nyquist frequency).
    const float max_frequency =
        static_cast<float>(context.loading_params.constants.default_sampling_rate) / 2.0F;

    // Initialize filterbank tensor and center-frequency buffer.
    context.filterbank =
        nn::Tensor(context.loading_params.audio_params.number_of_filters, number_of_bins);

    // Resize center-frequency array.
    context.center_frequencies.resize(context.loading_params.audio_params.number_of_filters + 2);

    // Compute filter center frequencies.
    for (int i = 0; i < context.loading_params.audio_params.number_of_filters + 2; ++i)
    {
        context.center_frequencies[i] =
            (max_frequency) * static_cast<float>(i) /
            static_cast<float>(context.loading_params.audio_params.number_of_filters + 1);
    }

    // Map center frequencies to FFT bin indices.
    for (size_t i = 0; i < context.center_frequencies.size(); ++i)
    {
        // Convert center frequency to FFT bin index.
        bin_indices[i] = static_cast<int>(
            floorf((static_cast<float>(fft_points) + 1.0F) * context.center_frequencies[i] /
                   static_cast<float>(context.loading_params.audio_params.target_sampling_rate)));
    }

    // Build triangular filters.
    for (int filter_index = 1;
        filter_index <= context.loading_params.audio_params.number_of_filters;
        ++filter_index)
    {
        // Previous bin index for current filter.
        const int previous_bin_index = bin_indices[filter_index - 1];

        // Center bin index for current filter.
        const int current_bin_index = bin_indices[filter_index];

        // Next bin index for current filter.
        const int next_bin_index = bin_indices[filter_index + 1];

        // Build ascending and descending sides of the triangular filter.
        if (current_bin_index > previous_bin_index)
        {
            // Ascending slope.
            for (int bin_index = previous_bin_index; bin_index < current_bin_index; ++bin_index)
            {
                context.filterbank(filter_index - 1, bin_index) =
                    static_cast<float>(bin_index - previous_bin_index) /
                    static_cast<float>(current_bin_index - previous_bin_index);
            }
        }
        // Descending slope.
        if (next_bin_index > current_bin_index)
        {
            // Descending slope.
            for (int bin_index = current_bin_index; bin_index < next_bin_index; ++bin_index)
            {
                context.filterbank(filter_index - 1, bin_index) =
                    static_cast<float>(next_bin_index - bin_index) /
                    static_cast<float>(next_bin_index - current_bin_index);
            }
        }
    }
}

/**
 * @brief Apply filterbank and log compression.
 *
 * Multiplies each frame power spectrum by the filterbank to obtain per-band
 * energies and then applies natural logarithm.
 * Formula: E = log(sum |X[k]|^2 * H_m[k]).
 *
 * @param power_spectrum Per-frame power spectra.
 * @param context Context containing parameters and filterbank.
 * @return Log energies per filter band.
 */
auto dot_power_filterbank(const nn::Tensor& power_spectrum, const PowerFilterbankConfig& context)
    -> nn::Tensor
{
    // Number of frames in the power spectrum.
    const long number_of_frames = static_cast<long>(power_spectrum.rows());

    // Number of filters in the filterbank.
    const long number_of_filters = static_cast<long>(context.filterbank.rows());

    // Number of frequency bins per frame.
    const long number_of_bins = static_cast<long>(power_spectrum.cols());

    // Output matrix for log-filterbank energies.
    nn::Tensor log_energies(number_of_frames, number_of_filters);

    // Compute log energies for each frame/filter pair.
    for (long frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Current frame index.
        for (long filter_index = 0; filter_index < number_of_filters; ++filter_index)
        {
            // Accumulator for weighted power-spectrum sum.
            float sum = 0.0F;
            for (long bin_index = 0; bin_index < number_of_bins; ++bin_index)
            {
                // Current frequency-bin index.
                sum += power_spectrum(frame_index, bin_index) *
                       context.filterbank(filter_index, bin_index);
            }
            sum = std::max(sum,
                context.loading_params.constants.min_log_energy); // Avoid log(0).
            log_energies(frame_index, filter_index) = logf(sum);
        }
    }
    return log_energies;
}

/**
 * @brief Compute the Discrete Cosine Transform (DCT-II).
 *
 * DCT decorrelates filterbank log energies, producing cepstral coefficients (LFCC).
 *
 * @param log_energies Filterbank log energies.
 * @param loading_params Loading and processing parameters.
 * @return LFCC cepstral coefficients.
 */
auto dct2(const nn::Tensor& log_energies, const LoadingAndProcessingParameters& loading_params)
    -> nn::Tensor
{
    // Number of input frames (energy vectors).
    const size_t number_of_frames = log_energies.rows();

    // Number of filters, equal to energies per frame.
    const size_t number_of_filters = log_energies.cols();

    // Output matrix for cepstral coefficients.
    nn::Tensor cepstral_coefficients(
        (long) number_of_frames, loading_params.audio_params.number_of_cepstrals);

    // Compute cepstral coefficients for each frame.
    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Compute cepstral terms for current frame.
        for (size_t cepstrum_index = 0;
            cepstrum_index < loading_params.audio_params.number_of_cepstrals;
            ++cepstrum_index)
        {
            // Accumulator for weighted DCT sum.
            float sum = 0.0F;

            // Compute current cepstral coefficient.
            for (long filter_index = 0; filter_index < static_cast<long>(number_of_filters);
                ++filter_index)
            {
                // Weighted DCT-II accumulation.
                sum += log_energies(    // Log energy
                           frame_index, // Frame index
                           filter_index // Filter index
                           ) *
                       cosf(std::numbers::pi_v<float> * static_cast<float>(cepstrum_index) *
                            (static_cast<float>(filter_index) +
                                loading_params.dct_config.filter_index_offset) /
                            static_cast<float>(number_of_filters) // Normalized filter index
                       );
            }

            // Orthogonal normalization.
            cepstral_coefficients(frame_index, cepstrum_index) =
                sum * sqrtf(loading_params.dct_config.normalization_factor_sqrt /
                            static_cast<float>(number_of_filters));

            // Extra normalization for the first cepstral coefficient.
            if (cepstrum_index == 0)
            {
                cepstral_coefficients(frame_index, cepstrum_index) *=
                    1.0F / std::numbers::sqrt2_v<float>;
            }
        }
    }
    return cepstral_coefficients;
}

/**
 * @brief Compute temporal deltas for feature vectors.
 * Deltas capture short-term dynamics of cepstral coefficients.
 *
 * @param features Feature matrix (LFCCs).
 * @param loading_params Loading and processing parameters.
 * @return Matrix with delta coefficients.
 */
auto compute_deltas(
    const nn::Tensor& features, const LoadingAndProcessingParameters& loading_params) -> nn::Tensor
{
    // Number of input frames.
    const long number_of_frames = features.rows();
    if (number_of_frames == 0)
    {
        // Return an explicitly empty tensor (0x0)
        return nn::Tensor(0, 0);
    }

    // Feature vector dimensionality.
    const size_t number_of_features = features.cols();

    // Precomputed denominator for delta formula.
    float denominator = 0.0F;

    // Output tensor for delta features.
    nn::Tensor delta_features(number_of_frames, (long) number_of_features);

    // Compute denominator term.
    for (int i = 1; i <= loading_params.audio_params.delta_window_span; ++i)
    {
        denominator += static_cast<float>(i) * static_cast<float>(i);
    }

    // Apply scaling factor used by the delta definition.
    denominator *= loading_params.delta_config.denominator_factor;

    // Compute deltas per frame and feature.
    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Iterate over feature dimensions.
        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            // Accumulator for numerator term.
            float numerator = 0.0F;

            // Build numerator for current feature delta.
            for (size_t delta_span = 1; delta_span <= loading_params.audio_params.delta_window_span;
                ++delta_span)
            {
                // Determine indices for c_{t+n} and c_{t-n} with boundary handling
                size_t index_plus_n = frame_index + delta_span;
                long index_minus_n = frame_index - delta_span;

                // Clamp indices to valid range
                if (index_plus_n >= number_of_frames)
                {
                    index_plus_n = number_of_frames - 1;
                }
                if (index_minus_n < 0)
                {
                    index_minus_n = 0;
                }

                // Temporal difference weighted by span distance.
                numerator +=
                    static_cast<float>(delta_span) * (features(index_plus_n, feature_index) -
                                                         features(index_minus_n, feature_index));
            }

            // Final delta value for frame/feature.
            delta_features(frame_index, feature_index) = numerator / denominator;
        }
    }

    // Return computed deltas.
    return delta_features;
}

// Windowing functions
auto hanning_window(int length) -> std::vector<double>
{
    if (length <= 0) return {};
    if (length == 1) return {1.0};
    std::vector<double> window(length);
    for (int i = 0; i < length; ++i)
    {
        window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (length - 1)));
    }
    return window;
}

auto apply_window(const std::vector<double>& signal, const std::vector<double>& window)
    -> std::vector<double>
{
    if (signal.size() != window.size())
    {
        throw std::invalid_argument("Signal and window must have the same size");
    }
    std::vector<double> result = signal;
    applyWindow(result, window);
    return result;
}

} // namespace nn::core::wave
