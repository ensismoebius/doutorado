#include "Experiment02Wavelets.hpp"

#include <string>
#include <vector>

#include "nn/wavelet/Types.h"
#include "nn/wavelet/waveletOperations.h"

auto get_wavelet_coeffs(const std::string& wavelet_name, const std::vector<double>& signal,
                        int max_level) -> wavelets::WaveletTransformResults
{
    if (wavelet_name == "Haar")
    {
        auto filter = wavelets::get_wavelet<wavelets::Haar>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub4")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub4>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub6")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub6>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub8")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub8>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub10")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub10>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub12")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub12>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub14")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub14>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub16")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub16>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub18")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub18>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub20")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub20>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub22")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub22>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub24")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub24>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub26")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub26>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub28")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub28>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub30")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub30>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub32")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub32>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub34")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub34>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub36")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub36>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub38")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub38>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub40")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub40>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub42")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub42>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub44")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub44>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    if (wavelet_name == "Daub46")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub46>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }

    auto filter = wavelets::get_wavelet<wavelets::Haar>();
    return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
}
