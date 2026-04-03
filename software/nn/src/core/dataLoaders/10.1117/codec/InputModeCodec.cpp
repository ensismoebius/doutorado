/**
 * @file src/core/dataLoaders/10.1117/codec/InputModeCodec.cpp
 * @brief Implementation for Inputmodecodec.
 *

 */

#include "nn/dataLoaders/10.1117/codec/InputModeCodec.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace
{

void toLowerAsciiInPlace(std::string& value)
{
    std::transform(value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
}

} // namespace

auto protocol101117InputModeToToken(Protocol101117InputMode mode) -> std::string
{
    switch (mode)
    {
        case Protocol101117InputMode::Concatenated:
            return "concatenated";
        case Protocol101117InputMode::EegOnly:
            return "eeg-only";
        case Protocol101117InputMode::AudioOnly:
            return "audio-only";
    }

    throw std::runtime_error("Unsupported input mode enum value");
}

auto parseProtocol101117InputModeToken(std::string token) -> Protocol101117InputMode
{
    toLowerAsciiInPlace(token);

    if (token == "concatenated")
    {
        return Protocol101117InputMode::Concatenated;
    }

    if (token == "eeg-only")
    {
        return Protocol101117InputMode::EegOnly;
    }

    if (token == "audio-only")
    {
        return Protocol101117InputMode::AudioOnly;
    }

    throw std::runtime_error("Unknown input mode: " + token);
}

auto supportedProtocol101117InputModeTokens() -> std::vector<std::string>
{
    return {"concatenated", "eeg-only", "audio-only"};
}
