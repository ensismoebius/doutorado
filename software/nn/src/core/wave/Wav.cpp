/**
 * @file Wav.cpp
 * @brief WAV file read/write implementation.
 */

/**
 * @author André Furlan <ensismoebius@gmail.com>
 *
 * Based on the original code of:
 * @author Rodrigo Capobianco Guido <guido@ieee.org>
 *
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 8 de ago de 2019
 */
#include "nn/wave/Wav.h"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using std::floor;
using std::pow;

const int FORMAT_KEY_MULTIPLIER = 10;
const int BITS_PER_BYTE = 8;

namespace
{

auto makeFormatKey(int wave_resolution, uint16_t channel_count) -> int
{
    return (wave_resolution * FORMAT_KEY_MULTIPLIER) + channel_count;
}

auto maxAmplitudeForBits(int bits_per_sample) -> double
{
    return floor((pow(2, bits_per_sample) - 1) / 2);
}

auto toPcm16Sample(double value, double max_amplitude) -> short
{
    // Accept both normalized input [-1, 1] and pre-scaled PCM input.
    const double scaled = (std::abs(value) <= 1.0) ? (value * max_amplitude) : value;

    const double min_short = static_cast<double>(std::numeric_limits<short>::min());
    const double max_short = static_cast<double>(std::numeric_limits<short>::max());
    const double clamped =
        (scaled < min_short) ? min_short : ((scaled > max_short) ? max_short : scaled);

    return static_cast<short>(std::llround(clamped));
}

} // namespace

Wav::Wav()
{
    reset_metadata();
}

Wav::Wav(uint32_t samplingRate,
    uint16_t bitsPerSample,
    uint16_t numOfChan,
    const double* audioData,
    size_t audioDataSize)
    : callbackFunction(nullptr)
{
    initialize_headers(samplingRate, bitsPerSample, numOfChan, audioDataSize);

    // Copy audio data
    this->data.assign(audioData, audioData + audioDataSize);
    this->amountOfData = audioDataSize;
    this->waveResolution = bitsPerSample;
}

void Wav::process()
{
    // The format key is a unique identifier created by combining the wave
    // resolution (e.g., 8 or 16 bits) and the number of channels. Multiplying
    // the resolution by 10 ensures that a unique value is generated for each
    // combination, allowing a simple switch statement to handle different audio
    // formats. For example, 8-bit mono is 81, while 16-bit mono is 161.
    int formatKey = makeFormatKey(waveResolution, this->header.numberOfChannels);

    switch (formatKey)
    {
        case Format8BitStereo:
        case Format16BitStereo:
            (*callbackFunction)(this->dataLeft, amountOfData, this->header.sampleRate, this->path);
            (*callbackFunction)(this->dataRight, amountOfData, this->header.sampleRate, this->path);
            break;
        case Format8BitMono:
        case Format16BitMono:
            (*callbackFunction)(this->data, amountOfData, this->header.sampleRate, this->path);
            break;
        default:
            throw std::runtime_error("Invalid number of channels and/or resolution");
            break;
    }
}

// flawfinder: ignore
void Wav::read(const std::string& _path)
{
    // Validate input path
    if (_path.empty())
    {
        throw std::invalid_argument("File path cannot be empty");
    }

    // Check if file exists and is accessible
    std::filesystem::path filePath(_path);
    if (!std::filesystem::exists(filePath))
    {
        throw std::runtime_error("File does not exist: " + _path);
    }
    if (!std::filesystem::is_regular_file(filePath))
    {
        throw std::runtime_error("Path is not a regular file: " + _path);
    }

    this->clear_vectors();
    this->reset_metadata();

    this->path = _path;

    std::ifstream ifs;
    ifs.open(path, std::ios::in | std::ios::binary); // flawfinder: ignore

    if (ifs.rdstate() != 0)
    {
        throw std::runtime_error("Reading the file " + path + " failed");
    }

    // Reads the file headers
    read_wave_headers(ifs);
    // Reads actual data
    read_wave_data(ifs);
    ifs.clear();
    ifs.close();
}

void Wav::write(const std::string& _path)
{
    path = _path;

    std::ofstream ofs;
    ofs.open(path, std::ios::out | std::ios::binary); // flawfinder: ignore

    if (!ofs.is_open())
    {
        std::cout << "Cannot open file: " << path;
        throw std::runtime_error("Impossible to open the file!");
    }

    // The format key is a unique identifier created by combining the wave
    // resolution (e.g., 8 or 16 bits) and the number of channels. Multiplying
    // the resolution by 10 ensures that a unique value is generated for each
    // combination, allowing a simple switch statement to handle different audio
    // formats. For example, 8-bit mono is 81, while 16-bit mono is 161.
    int formatKey = makeFormatKey(waveResolution, this->header.numberOfChannels);

    switch (formatKey)
    {
        case Format8BitMono:
            write8BitMono(ofs);
            break;
        case Format8BitStereo:
            write8BitStereo(ofs);
            break;
        case Format16BitMono:
            write16BitMono(ofs);
            break;
        case Format16BitStereo:
            write16BitStereo(ofs);
            break;
        default:
            ofs.close();
            throw std::runtime_error("Invalid number of channels and/or resolution");
            break;
    }

    ofs.close();
}

void Wav::write(const std::string& _path, const std::vector<float>& inputData, int samplingRate)
{
    path = _path;

    std::ofstream ofs;
    ofs.open(path, std::ios::out | std::ios::binary); // flawfinder: ignore

    if (!ofs.is_open())
    {
        std::cout << "Cannot open file: " << path;
        throw std::runtime_error("Impossible to open the file!");
    }

    const uint16_t bitsPerSample = 16;
    const uint16_t numberOfChannels = 1;
    initialize_headers(samplingRate, bitsPerSample, numberOfChannels, inputData.size());

    this->amountOfData = inputData.size();
    this->waveResolution = bitsPerSample;

    // Write header
    Wav::write_binary(ofs, this->header);

    // Calculate maximum amplitude for the audio format
    const float maxAmplitude = static_cast<float>(maxAmplitudeForBits(this->waveResolution));

    // Convert float data to short and write to file
    for (float value : inputData)
    {
        const auto sample = toPcm16Sample(value, maxAmplitude);
        Wav::write_binary(ofs, sample);
    }

    ofs.close();
}

void Wav::write(
    const std::string& _path, const std::vector<std::vector<float>>& inputData, int samplingRate)
{
    path = _path;

    std::ofstream ofs;
    ofs.open(path, std::ios::out | std::ios::binary); // flawfinder: ignore

    if (!ofs.is_open())
    {
        std::cout << "Cannot open file: " << path;
        throw std::runtime_error("Impossible to open the file!");
    }

    size_t numberOfChannels = inputData.size();
    if (numberOfChannels == 0)
    {
        throw std::runtime_error("Input data vector is empty.");
    }
    size_t numSamples = inputData[0].size();

    if (numberOfChannels > 2)
    {
        throw std::runtime_error(
            "Only mono (1 channel) or stereo (2 channels) are supported for "
            "std::vector<std::vector<float>> input.");
    }

    const uint16_t bitsPerSample = 16;
    initialize_headers(
        samplingRate, bitsPerSample, static_cast<uint16_t>(numberOfChannels), numSamples);

    this->amountOfData = numSamples;
    this->waveResolution = bitsPerSample;

    // Write header
    Wav::write_binary(ofs, this->header);

    // Calculate maximum amplitude for the audio format
    const float maxAmplitude = static_cast<float>(maxAmplitudeForBits(this->waveResolution));

    if (numberOfChannels == 1)
    {
        for (size_t i = 0; i < numSamples; ++i)
        {
            const auto sample = toPcm16Sample(inputData[0][i], maxAmplitude);
            Wav::write_binary(ofs, sample);
        }
    }
    else // numberOfChannels == 2
    {
        for (size_t i = 0; i < numSamples; ++i)
        {
            const auto left_sample = toPcm16Sample(inputData[0][i], maxAmplitude);
            const auto right_sample = toPcm16Sample(inputData[1][i], maxAmplitude);
            Wav::write_binary(ofs, left_sample);
            Wav::write_binary(ofs, right_sample);
        }
    }

    ofs.close();
}

auto Wav::get_data() const -> const std::vector<double>&
{
    return data;
}

auto Wav::get_data_left() const -> const std::vector<double>&
{
    return dataLeft;
}

auto Wav::get_data_right() const -> const std::vector<double>&
{
    return dataRight;
}

auto Wav::get_path() const -> std::string
{
    return this->path;
}

void Wav::set_callback_function(     //
    void (*_callbackFunction)(       //
        std::vector<double>& signal, //
        size_t& signalLength,        //
        uint32_t samplingRate,       //
        std::string path             //
        )                            // callback function
)
{
    this->callbackFunction = _callbackFunction;
}

void Wav::read_wave_data(std::ifstream& ifs)
{
    // The format key is a unique identifier created by combining the wave
    // resolution (e.g., 8 or 16 bits) and the number of channels. Multiplying
    // the resolution by 10 ensures that a unique value is generated for each
    // combination, allowing a simple switch statement to handle different audio
    // formats. For example, 8-bit mono is 81, while 16-bit mono is 161.
    int formatKey = makeFormatKey(waveResolution, this->header.numberOfChannels);

    switch (formatKey)
    {
        case Format8BitMono:
            read8BitMono(ifs);
            break;
        case Format8BitStereo:
            read8BitStereo(ifs);
            break;
        case Format16BitMono:
            read16BitMono(ifs);
            break;
        case Format16BitStereo:
            read16BitStereo(ifs);
            break;
        default:
            ifs.close();
            throw std::runtime_error("Invalid number of channels and/or resolution");
            break;
    }
}

void Wav::read_wave_headers(std::ifstream& ifs)
{
    ifs.seekg(0, std::ios::beg);
    Wav::read_binary(ifs, this->header);

    if (this->header.audioFormat != 1)
    {
        throw std::runtime_error(this->path + " not in PCM format!");
    }

    waveResolution = static_cast<int>((this->header.byteRate * BITS_PER_BYTE) /
                                      (this->header.numberOfChannels * this->header.sampleRate));
    amountOfData = this->header.dataSubchunkSize / this->header.blockAlign;

    // Validate that the file is large enough to contain the expected data
    ifs.seekg(0, std::ios::end);
    std::streamsize fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (fileSize < static_cast<std::streamsize>(44 + this->header.dataSubchunkSize))
    {
        throw std::runtime_error("File is too small to contain the expected audio data");
    }
}

inline void Wav::write8BitMono(std::ofstream& ofs)
{
    Wav::write_binary(ofs, this->header);

    for (size_t i = 0; i < amountOfData; i++)
    {
        unsigned char waveformdata = static_cast<unsigned char>(this->data.at(i));
        Wav::write_binary(ofs, waveformdata);
    }
}

inline void Wav::write8BitStereo(std::ofstream& ofs)
{
    Wav::write_binary(ofs, this->header);

    for (size_t i = 0; i < amountOfData; i++)
    {
        unsigned char waveformdata_left = static_cast<unsigned char>(this->dataLeft.at(i));
        Wav::write_binary(ofs, waveformdata_left);
        unsigned char waveformdata_right = static_cast<unsigned char>(this->dataRight.at(i));
        Wav::write_binary(ofs, waveformdata_right);
    }
}

inline void Wav::write16BitMono(std::ofstream& ofs)
{
    Wav::write_binary(ofs, this->header);

    unsigned char waveformdata_lsb = 0;
    unsigned char waveformdata_msb = 0;

    const double maxAmplitude = maxAmplitudeForBits(this->waveResolution);

    for (size_t i = 0; i < amountOfData; i++)
    {
        const auto sample = toPcm16Sample(this->data.at(i), maxAmplitude);
        split_16bit_to_8bit(sample, &waveformdata_lsb, &waveformdata_msb);
        Wav::write_binary(ofs, waveformdata_lsb);
        Wav::write_binary(ofs, waveformdata_msb);
    }
}

inline void Wav::write16BitStereo(std::ofstream& ofs)
{
    Wav::write_binary(ofs, this->header);

    unsigned char waveformdata_lsb_left = 0;
    unsigned char waveformdata_lsb_right = 0;
    unsigned char waveformdata_msb_left = 0;
    unsigned char waveformdata_msb_right = 0;

    const double maxAmplitude = maxAmplitudeForBits(this->waveResolution);

    for (size_t i = 0; i < amountOfData; i++)
    {
        const auto left_sample = toPcm16Sample(this->dataLeft.at(i), maxAmplitude);
        split_16bit_to_8bit(left_sample, &waveformdata_lsb_left, &waveformdata_msb_left);
        const auto right_sample = toPcm16Sample(this->dataRight.at(i), maxAmplitude);
        split_16bit_to_8bit(right_sample, &waveformdata_lsb_right, &waveformdata_msb_right);
        Wav::write_binary(ofs, waveformdata_lsb_left);
        Wav::write_binary(ofs, waveformdata_msb_left);
        Wav::write_binary(ofs, waveformdata_lsb_right);
        Wav::write_binary(ofs, waveformdata_msb_right);
    }
}

inline void Wav::read8BitMono(std::ifstream& ifs)
{
    unsigned char waveformdata = 0;
    this->data.resize(amountOfData, 0);
    for (size_t i = 0; i < amountOfData; i++)
    {
        if (!ifs.good())
        {
            throw std::runtime_error("Error reading audio data: unexpected end of file");
        }

        Wav::read_binary(ifs, waveformdata);

        if (i >= this->data.size())
        {
            throw std::runtime_error("Buffer overflow detected in read8BitMono");
        }

        this->data.at(i) = static_cast<double>(waveformdata);
    }
}

inline void Wav::read8BitStereo(std::ifstream& ifs)
{
    unsigned char waveformdata_right = 0;
    unsigned char waveformdata_left = 0;
    this->dataLeft.resize(amountOfData, 0);
    this->dataRight.resize(amountOfData, 0);
    for (size_t i = 0; i < amountOfData; i++)
    {
        if (!ifs.good())
        {
            throw std::runtime_error("Error reading audio data: unexpected end of file");
        }

        Wav::read_binary(ifs, waveformdata_left);
        Wav::read_binary(ifs, waveformdata_right);

        if (i >= this->dataLeft.size() || i >= this->dataRight.size())
        {
            throw std::runtime_error("Buffer overflow detected in read8BitStereo");
        }

        this->dataLeft.at(i) = static_cast<double>(waveformdata_right);
        this->dataRight.at(i) = static_cast<double>(waveformdata_left);
    }
}

inline void Wav::read16BitMono(std::ifstream& ifs)
{
    unsigned char waveformdata_lsb = 0;
    unsigned char waveformdata_msb = 0;
    this->data.resize(amountOfData, 0);
    for (size_t i = 0; i < amountOfData; i++)
    {
        if (!ifs.good())
        {
            throw std::runtime_error("Error reading audio data: unexpected end of file");
        }

        Wav::read_binary(ifs, waveformdata_lsb);
        Wav::read_binary(ifs, waveformdata_msb);

        if (i >= this->data.size())
        {
            throw std::runtime_error("Buffer overflow detected in read16BitMono");
        }

        this->data.at(i) =
            static_cast<double>(combine_8bit_to_16bit(waveformdata_lsb, waveformdata_msb));
    }
}

inline void Wav::read16BitStereo(std::ifstream& ifs)
{
    unsigned char waveformdata_lsb_left = 0;
    unsigned char waveformdata_lsb_right = 0;
    unsigned char waveformdata_msb_left = 0;
    unsigned char waveformdata_msb_right = 0;
    this->dataLeft.resize(amountOfData, 0);
    this->dataRight.resize(amountOfData, 0);
    for (size_t i = 0; i < amountOfData; i++)
    {
        if (!ifs.good())
        {
            throw std::runtime_error("Error reading audio data: unexpected end of file");
        }
        Wav::read_binary(ifs, waveformdata_lsb_left);
        Wav::read_binary(ifs, waveformdata_msb_left);
        Wav::read_binary(ifs, waveformdata_lsb_right);
        Wav::read_binary(ifs, waveformdata_msb_right);

        if (i >= this->dataLeft.size() || i >= this->dataRight.size())
        {
            throw std::runtime_error("Buffer overflow detected in read16BitStereo");
        }

        this->dataLeft.at(i) = static_cast<double>(
            combine_8bit_to_16bit(waveformdata_lsb_left, waveformdata_msb_left));

        this->dataRight.at(i) = static_cast<double>(
            combine_8bit_to_16bit(waveformdata_lsb_right, waveformdata_msb_right));
    }
}

auto Wav::combine_8bit_to_16bit(unsigned char lsb, unsigned char msb) -> short
{
    // Combine the most significant byte (msb) and the least significant byte (lsb)
    // into a single 16-bit signed short. This is achieved by shifting the msb
    // 8 bits to the left (to occupy the upper 8 bits of the short) and then
    // performing a bitwise OR with the lsb (which occupies the lower 8 bits).
    // This assumes a little-endian system where the lsb comes first.
    return static_cast<short>((static_cast<short>(msb) << 8) | lsb);
}

void Wav::split_16bit_to_8bit(short sample, unsigned char* lsb, unsigned char* msb)
{
    // Extract the least significant byte (lsb) from the 16-bit sample.
    // This is done by applying a bitwise AND with a mask of 0xFF, which
    // isolates the lower 8 bits of the sample.
    *lsb = sample & 0xFF;

    // Extract the most significant byte (msb) from the 16-bit sample.
    // This is done by shifting the sample 8 bits to the right, which moves
    // the upper 8 bits into the lower 8 bits position, and then applying a
    // bitwise AND with a mask of 0xFF to isolate them.
    *msb = (sample >> 8) & 0xFF;
}

void Wav::clear_vectors()
{
    this->data.clear();
    this->dataLeft.clear();
    this->dataRight.clear();
}

void Wav::reset_metadata()
{
    this->amountOfData = 0;
    this->waveResolution = 0;
    this->header = {};
}

void Wav::initialize_headers(
    uint32_t samplingRate, uint16_t bitsPerSample, uint16_t numberOfChannels, size_t numSamples)
{
    this->header.riffChunkId = {'R', 'I', 'F', 'F'};
    this->header.waveFormat = {'W', 'A', 'V', 'E'};
    this->header.fmtSubchunkId = {'f', 'm', 't', ' '};
    this->header.fmtSubchunkSize = 16; // PCM
    this->header.audioFormat = 1;      // PCM
    this->header.numberOfChannels = numberOfChannels;
    this->header.sampleRate = samplingRate;
    this->header.bitsPerSample = bitsPerSample;
    this->header.byteRate = samplingRate * numberOfChannels * (bitsPerSample / 8);
    this->header.blockAlign = numberOfChannels * (bitsPerSample / 8);
    this->header.dataSubchunkId = {'d', 'a', 't', 'a'};
    this->header.dataSubchunkSize = numSamples * numberOfChannels * (bitsPerSample / 8);
    this->header.riffChunkSize = 36 + this->header.dataSubchunkSize;
}