/**
 * @file src/experiments/02/Experiment02Wavelets.cpp
 * @brief Implementation for Experiment02wavelets.
 *

 */

#include "Experiment02Wavelets.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/wavelet/Types.h"
#include "nn/wavelet/waveletOperations.h"

namespace
{
template <typename TWavelet>
auto compute_packet_transform(const std::vector<double>& signal, int max_level)
    -> wavelets::WaveletTransformResults
{
    auto filter = wavelets::get_wavelet<TWavelet>();
    return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
}

using WaveletTransformFn = wavelets::WaveletTransformResults (*)(const std::vector<double>&, int);

struct WaveletDispatchEntry
{
    const char* wavelet_name;
    WaveletTransformFn transform_fn;
};

constexpr std::array<WaveletDispatchEntry, 23> kWaveletDispatch = {{
    {"Haar", &compute_packet_transform<wavelets::Haar>},
    {"Daub4", &compute_packet_transform<wavelets::Daub4>},
    {"Daub6", &compute_packet_transform<wavelets::Daub6>},
    {"Daub8", &compute_packet_transform<wavelets::Daub8>},
    {"Daub10", &compute_packet_transform<wavelets::Daub10>},
    {"Daub12", &compute_packet_transform<wavelets::Daub12>},
    {"Daub14", &compute_packet_transform<wavelets::Daub14>},
    {"Daub16", &compute_packet_transform<wavelets::Daub16>},
    {"Daub18", &compute_packet_transform<wavelets::Daub18>},
    {"Daub20", &compute_packet_transform<wavelets::Daub20>},
    {"Daub22", &compute_packet_transform<wavelets::Daub22>},
    {"Daub24", &compute_packet_transform<wavelets::Daub24>},
    {"Daub26", &compute_packet_transform<wavelets::Daub26>},
    {"Daub28", &compute_packet_transform<wavelets::Daub28>},
    {"Daub30", &compute_packet_transform<wavelets::Daub30>},
    {"Daub32", &compute_packet_transform<wavelets::Daub32>},
    {"Daub34", &compute_packet_transform<wavelets::Daub34>},
    {"Daub36", &compute_packet_transform<wavelets::Daub36>},
    {"Daub38", &compute_packet_transform<wavelets::Daub38>},
    {"Daub40", &compute_packet_transform<wavelets::Daub40>},
    {"Daub42", &compute_packet_transform<wavelets::Daub42>},
    {"Daub44", &compute_packet_transform<wavelets::Daub44>},
    {"Daub46", &compute_packet_transform<wavelets::Daub46>},
}};
} // namespace

auto get_wavelet_coeffs(
    const std::string& wavelet_name, const std::vector<double>& signal, int max_level)
    -> wavelets::WaveletTransformResults
{
    if (signal.empty())
    {
        throw std::invalid_argument("Signal size must be greater than zero.");
    }

    std::vector<double> padded_signal(signal.begin(), signal.end());
    const int padded_size = wavelets::get_next_power_of_two(static_cast<double>(signal.size()));
    if (padded_size > static_cast<int>(signal.size()))
    {
        padded_signal.resize(static_cast<std::size_t>(padded_size), 0.0);
    }

    const auto dispatch_it = std::find_if(kWaveletDispatch.begin(),
        kWaveletDispatch.end(),
        [&wavelet_name](const auto& dispatch_entry)
        { return wavelet_name == dispatch_entry.wavelet_name; });

    if (dispatch_it != kWaveletDispatch.end())
    {
        return dispatch_it->transform_fn(padded_signal, max_level);
    }

    return compute_packet_transform<wavelets::Haar>(padded_signal, max_level);
}
