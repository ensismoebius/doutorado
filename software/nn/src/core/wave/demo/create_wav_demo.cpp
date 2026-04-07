/**
 * @file create_wav_demo.cpp
 * @brief Tiny demo that synthesizes a sine wave and writes/reads a WAV file.
 */

#include <cmath>
#include <filesystem>
#include <string>

#include "../Wav.h"
#include "../filter_operations.hpp"
#include "nn/logging/Logger.hpp"

using std::sin;

auto main() -> int
{
    Wav wavFile;
    std::vector<float> data;

    // Generate a simple sine wave
    const float frequency = 440.0; // Hz
    const float sampleRate = 44100.0;
    const float duration = 5.0F; // seconds
    const std::string filePath = std::filesystem::temp_directory_path().string() + "/sine_wave.wav";

    for (float i = 0; i < sampleRate * duration; i++)
    {
        float t = i / sampleRate;
        data.push_back((float) sin(2.0F * M_PI * frequency * t));
    }

    // This is used for nothing but to test the createAlpha function
    double alpha = createAlpha(sampleRate, 2000.0);

    NN_LOG_INFO(
        "Alpha value for " + std::to_string(alpha) + " for 2000 Hz cutoff at 44100 Hz sample rate");

    wavFile.write(filePath, data, static_cast<int>(sampleRate));
    wavFile.read(filePath); // flawfinder: ignore

    auto readData = wavFile.getData();

    // Simple verification
    if (readData.size() != data.size())
    {
        NN_LOG_ERROR("Test failed: Data size mismatch!");
        return 1;
    }
    return 0;
}