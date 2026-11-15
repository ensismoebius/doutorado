#include "DegradeVoices.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "VoiceDegradation.hpp"
#include "VoiceDysphonia.hpp"

namespace thesis
{

void degrade_voices(ThesisDatasetView& view, const ThesisConfig::Degradation& dg)
{
    const double voice_rate_hz = require_rate(view.voice_sample_rate, "voice");
    const bool is_whisper = dg.profile == "whisper_mix";
    DysphoniaParams f0_params;
    double native_rate_hz = 0.0;
    if (!is_whisper)
    {
        f0_params = dysphonia_params(dg);
        native_rate_hz = require_rate(view.voice_native_rate, "native voice");
    }
    const auto n = static_cast<long>(view.samples.size());
    std::exception_ptr first_error;
    long first_error_sample = -1;
#pragma omp parallel for schedule(dynamic, 8)
    for (long i = 0; i < n; ++i)
    {
        bool failed;
#pragma omp critical(degrade_voices_error)
        failed = first_error != nullptr;
        if (failed) continue;
        try
        {
            nn::Tensor& audio = view.samples[static_cast<std::size_t>(i)].audio;
            const float* src = audio.data_ptr();
            const std::vector<float> x(src, src + audio.size());
            const std::uint32_t seed =
                dg.seed.get() ^ (static_cast<std::uint32_t>(i) * 2654435761U);
            const std::vector<float> y =
                is_whisper ? degrade_voice(x, voice_rate_hz, dg.level, seed)
                           : degrade_voice_f0(x, voice_rate_hz, native_rate_hz, f0_params, seed);
            if (y.size() != x.size())
                throw std::runtime_error("degradation returned " + std::to_string(y.size()) +
                                         " samples for an input of " + std::to_string(x.size()));
            std::copy(y.begin(), y.end(), audio.mutable_data_ptr());
        }
        catch (...)
        {
#pragma omp critical(degrade_voices_error)
            if (first_error == nullptr)
            {
                first_error = std::current_exception();
                first_error_sample = i;
            }
        }
    }
    if (first_error != nullptr)
    {
        try
        {
            std::rethrow_exception(first_error);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("degradation of sample " + std::to_string(first_error_sample) +
                                     " of " + std::to_string(n) + " failed: " + e.what());
        }
    }
}

} // namespace thesis
