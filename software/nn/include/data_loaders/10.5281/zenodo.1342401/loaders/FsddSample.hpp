#pragma once
// FsddSample.hpp — Per-file metadata parsed from an FSDD WAV filename.

#include <string>

namespace nn::dataLoaders::fsdd
{

// Metadata decoded from an FSDD filename stem (e.g. "0_jackson_3").
struct FsddFileInfo
{
    int         digit;    // 0–9
    std::string speaker;  // e.g. "jackson"
    int         trial;    // 0–49
};

} // namespace nn::dataLoaders::fsdd
