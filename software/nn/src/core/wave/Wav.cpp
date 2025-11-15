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
#include "Wav.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using std::floor;
using std::pow;

Wav::Wav()
{
    resetMetaData();
}

Wav::Wav(uint32_t samplingRate, uint16_t bitsPerSample, uint16_t numOfChan, const double* audioData,
         size_t audioDataSize)
{
    initializeHeaders(samplingRate, bitsPerSample, numOfChan, audioDataSize);

    // Copy audio data
    this->data.assign(audioData, audioData + audioDataSize);
    this->amountOfData = audioDataSize;
    this->waveResolution = bitsPerSample;
}

void Wav::process()
{
    if (callbackFunction == nullptr)
    {
        return;
    }

    int resPlusCha = (waveResolution * 10) + this->headers.numOfChan;

    switch (resPlusCha)
    {
        case 82:
        case 162:
            (*callbackFunction)(
                this->dataLeft, amountOfData, this->headers.samplingrate, this->path);
            (*callbackFunction)(
                this->dataRight, amountOfData, this->headers.samplingrate, this->path);
            break;
        case 81:
        case 161:
            (*callbackFunction)(this->data, amountOfData, this->headers.samplingrate, this->path);
            break;
        default:
            throw std::runtime_error("Invalid number of channels and/or resolution");
            break;
    }
}

void Wav::read(const std::string& _path)
{
    this->clearVectors();
    this->resetMetaData();

    this->path = _path;

    std::ifstream ifs;
    ifs.open(path, std::ios::in | std::ios::binary);

    if (ifs.rdstate() != 0)
    {
        throw std::runtime_error("Reading the file " + path + " failed");
    }

    // Reads the file headers
    readWaveHeaders(ifs);
    // Reads actual data
    readWaveData(ifs);
    ifs.clear();
    ifs.close();
}

void Wav::write(const std::string& _path)
{
    path = _path;

    std::ofstream ofs;
    ofs.open(path, std::ios::out | std::ios::binary);

    if (!ofs.is_open())
    {
        std::cout << "Cannot open file: " << path;
        throw std::runtime_error("Impossible to open the file!");
        return;
    }

    int resPlusCha = (waveResolution * 10) + this->headers.numOfChan;

    switch (resPlusCha)
    {
        case 81:
            write8BitMono(ofs);
            break;
        case 82:
            write8BitStereo(ofs);
            break;
        case 161:
            write16BitMono(ofs);
            break;
        case 162:
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
    ofs.open(path, std::ios::out | std::ios::binary);

    if (!ofs.is_open())
    {
        std::cout << "Cannot open file: " << path;
        throw std::runtime_error("Impossible to open the file!");
    }

    const uint16_t bitsPerSample = 16;
    const uint16_t numChannels = 1;
    initializeHeaders(samplingRate, bitsPerSample, numChannels, inputData.size());

    this->amountOfData = inputData.size();
    this->waveResolution = bitsPerSample;

    // Write header
    ofs.write((char*) (&this->headers), sizeof(this->headers));

    // Calculate maximum amplitude for the audio format
    const float maxAmplitude = floor((pow(2, this->waveResolution) - 1) / 2);

    // Convert float data to short and write to file
    for (float i : inputData)
    {
        auto sample = static_cast<short>(i * maxAmplitude);
        ofs.write((char*) (&sample), sizeof(sample));
    }

    ofs.close();
}

void Wav::write(const std::string& _path, const std::vector<std::vector<float>>& inputData,
                int samplingRate)
{
    path = _path;

    std::ofstream ofs;
    ofs.open(path, std::ios::out | std::ios::binary);

    if (!ofs.is_open())
    {
        std::cout << "Cannot open file: " << path;
        throw std::runtime_error("Impossible to open the file!");
    }

    size_t numOfChannels = inputData.size();
    if (numOfChannels == 0)
    {
        throw std::runtime_error("Input data vector is empty.");
    }
    size_t numSamples = inputData[0].size();

    if (numOfChannels < 1 || numOfChannels > 2)
    {
        throw std::runtime_error(
            "Only mono (1 channel) or stereo (2 channels) are supported for "
            "std::vector<std::vector<float>> input.");
    }

    const uint16_t bitsPerSample = 16;
    initializeHeaders(samplingRate, bitsPerSample, numOfChannels, numSamples);

    this->amountOfData = numSamples;
    this->waveResolution = bitsPerSample;

    // Write header
    ofs.write((char*) (&this->headers), sizeof(this->headers));

    // Calculate maximum amplitude for the audio format
    const float maxAmplitude = floor((pow(2, this->waveResolution) - 1) / 2);

    if (numOfChannels == 1)
    {
        for (size_t i = 0; i < numSamples; ++i)
        {
            auto sample = static_cast<short>(inputData[0][i] * maxAmplitude);
            ofs.write((char*) (&sample), sizeof(sample));
        }
    }
    else // numOfChannels == 2
    {
        for (size_t i = 0; i < numSamples; ++i)
        {
            auto left_sample = static_cast<short>(inputData[0][i] * maxAmplitude);
            auto right_sample = static_cast<short>(inputData[1][i] * maxAmplitude);
            ofs.write((char*) (&left_sample), sizeof(left_sample));
            ofs.write((char*) (&right_sample), sizeof(right_sample));
        }
    }

    ofs.close();
}

auto Wav::getData() const -> std::vector<double>
{
    return data;
}

auto Wav::getDataLeft() const -> std::vector<double>
{
    return dataLeft;
}

auto Wav::getDataRight() const -> std::vector<double>
{
    return dataRight;
}

auto Wav::getPath() const -> std::string
{
    return this->path;
}

void Wav::setCallbackFunction(       //
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

void Wav::readWaveData(std::ifstream& ifs)
{
    int resPlusCha = (waveResolution * 10) + this->headers.numOfChan;

    switch (resPlusCha)
    {
        case 81:
            read8BitMono(ifs);
            break;
        case 82:
            read8BitStereo(ifs);
            break;
        case 161:
            read16BitMono(ifs);
            break;
        case 162:
            read16BitStereo(ifs);
            break;
        default:
            ifs.close();
            throw std::runtime_error("Invalid number of channels and/or resolution");
            break;
    }
}

void Wav::readWaveHeaders(std::ifstream& ifs)
{
    ifs.seekg(0, std::ios::beg);
    ifs.read((char*) &this->headers, sizeof(this->headers));

    if (this->headers.audioFormat != 1)
    {
        throw std::runtime_error(this->path + " not in PCM format!");
        return;
    }

    waveResolution =
        (this->headers.bytesPerSec * 8) / (this->headers.numOfChan * this->headers.samplingrate);
    amountOfData = this->headers.subchunk2Size / this->headers.blockAlign;
}

inline void Wav::write8BitMono(std::ofstream& ofs)
{
    ofs.write((char*) (&this->headers), sizeof(this->headers));

    unsigned char waveformdata;

    for (int i = 0; i < amountOfData; i++)
    {
        waveformdata = (unsigned char) (this->data.at(i));
        ofs.write((char*) (&waveformdata), sizeof(waveformdata));
    }
}

inline void Wav::write8BitStereo(std::ofstream& ofs)
{
    ofs.write((char*) (&this->headers), sizeof(this->headers));

    unsigned char waveformdata_right;
    unsigned char waveformdata_left;

    for (int i = 0; i < amountOfData; i++)
    {
        waveformdata_left = (unsigned char) (this->dataLeft.at(i));
        ofs.write((char*) (&waveformdata_left), sizeof(waveformdata_left));
        waveformdata_right = (unsigned char) (this->dataRight.at(i));
        ofs.write((char*) (&waveformdata_right), sizeof(waveformdata_right));
    }
}

inline void Wav::write16BitMono(std::ofstream& ofs)
{
    ofs.write((char*) (&this->headers), sizeof(this->headers));

    unsigned char waveformdata_lsb;
    unsigned char waveformdata_msb;

    const double maxAmplitude = floor((pow(2, this->waveResolution) - 1) / 2);

    for (int i = 0; i < amountOfData; i++)
    {
        const auto sample = static_cast<short>(this->data.at(i) * maxAmplitude);
        split16BitTo8Bit(sample, &waveformdata_lsb, &waveformdata_msb);
        ofs.write((char*) (&waveformdata_lsb), sizeof(waveformdata_lsb));
        ofs.write((char*) (&waveformdata_msb), sizeof(waveformdata_msb));
    }
}

inline void Wav::write16BitStereo(std::ofstream& ofs)
{
    ofs.write((char*) &this->headers, sizeof(this->headers));

    unsigned char waveformdata_lsb_left;
    unsigned char waveformdata_lsb_right;
    unsigned char waveformdata_msb_left;
    unsigned char waveformdata_msb_right;

    const double maxAmplitude = floor((pow(2, this->waveResolution) - 1) / 2);

    for (int i = 0; i < amountOfData; i++)
    {
        const auto left_sample = static_cast<short>(this->dataLeft.at(i) * maxAmplitude);
        split16BitTo8Bit(left_sample, &waveformdata_lsb_left, &waveformdata_msb_left);
        const auto right_sample = static_cast<short>(this->dataRight.at(i) * maxAmplitude);
        split16BitTo8Bit(right_sample, &waveformdata_lsb_right, &waveformdata_msb_right);
        ofs.write((char*) (&waveformdata_lsb_left), sizeof(waveformdata_lsb_left));
        ofs.write((char*) (&waveformdata_msb_left), sizeof(waveformdata_msb_left));
        ofs.write((char*) (&waveformdata_lsb_right), sizeof(waveformdata_lsb_right));
        ofs.write((char*) (&waveformdata_msb_right), sizeof(waveformdata_msb_right));
    }
}

inline void Wav::read8BitMono(std::ifstream& ifs)
{
    unsigned char waveformdata;
    this->data.resize(amountOfData, 0);
    for (int i = 0; i < amountOfData; i++)
    {
        ifs.read((char*) (&waveformdata), sizeof(waveformdata));
        this->data.at(i) = (double) ((waveformdata));
    }
}

inline void Wav::read8BitStereo(std::ifstream& ifs)
{
    unsigned char waveformdata_right;
    unsigned char waveformdata_left;
    this->dataLeft.resize(amountOfData, 0);
    this->dataRight.resize(amountOfData, 0);
    for (int i = 0; i < amountOfData; i++)
    {
        ifs.read((char*) (&waveformdata_left), sizeof(waveformdata_left));
        ifs.read((char*) (&waveformdata_right), sizeof(waveformdata_right));
        this->dataLeft.at(i) = (double) ((waveformdata_right));
        this->dataRight.at(i) = (double) ((waveformdata_left));
    }
}

inline void Wav::read16BitMono(std::ifstream& ifs)
{
    unsigned char waveformdata_lsb;
    unsigned char waveformdata_msb;
    this->data.resize(amountOfData, 0);
    for (int i = 0; i < amountOfData; i++)
    {
        ifs.read((char*) (&waveformdata_lsb), sizeof(waveformdata_lsb));
        ifs.read((char*) (&waveformdata_msb), sizeof(waveformdata_msb));
        this->data.at(i) = (double) ((combine8BitTo16Bit(waveformdata_lsb, waveformdata_msb)));
    }
}

inline void Wav::read16BitStereo(std::ifstream& ifs)
{
    unsigned char waveformdata_lsb_left;
    unsigned char waveformdata_lsb_right;
    unsigned char waveformdata_msb_left;
    unsigned char waveformdata_msb_right;
    this->dataLeft.resize(amountOfData, 0);
    this->dataRight.resize(amountOfData, 0);
    for (int i = 0; i < amountOfData; i++)
    {
        ifs.read((char*) (&waveformdata_lsb_left), sizeof(waveformdata_lsb_left));
        ifs.read((char*) (&waveformdata_msb_left), sizeof(waveformdata_msb_left));
        ifs.read((char*) (&waveformdata_lsb_right), sizeof(waveformdata_lsb_right));
        ifs.read((char*) (&waveformdata_msb_right), sizeof(waveformdata_msb_right));
        this->dataLeft.at(i) =
            (double) ((combine8BitTo16Bit(waveformdata_lsb_left, waveformdata_msb_left)));
        this->dataRight.at(i) =
            (double) ((combine8BitTo16Bit(waveformdata_lsb_right, waveformdata_msb_right)));
    }
}

auto Wav::combine8BitTo16Bit(unsigned char lsb, unsigned char msb) -> short
{
    // Combine the most significant byte (msb) and the least significant byte (lsb)
    // into a single 16-bit signed short. This is achieved by shifting the msb
    // 8 bits to the left (to occupy the upper 8 bits of the short) and then
    // performing a bitwise OR with the lsb (which occupies the lower 8 bits).
    // This assumes a little-endian system where the lsb comes first.
    return (static_cast<short>(msb) << 8) | lsb;
}

void Wav::split16BitTo8Bit(short sample, unsigned char* lsb, unsigned char* msb)
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

void Wav::clearVectors()
{
    this->data.clear();
    this->dataLeft.clear();
    this->dataRight.clear();
}

void Wav::resetMetaData()
{
    this->amountOfData = 0;
    this->waveResolution = 0;
    /* RIFF Chunk Descriptor */
    this->headers.RIFF[0] = '\0';
    this->headers.RIFF[1] = '\0';
    this->headers.RIFF[2] = '\0';
    this->headers.RIFF[3] = '\0';
    this->headers.chunkSize = 0;
    this->headers.WAVE[0] = '\0';
    this->headers.WAVE[1] = '\0';
    this->headers.WAVE[2] = '\0';
    this->headers.WAVE[3] = '\0';
    /* "fmt" sub-chunk */
    this->headers.fmt[0] = '\0';
    this->headers.fmt[1] = '\0';
    this->headers.fmt[2] = '\0';
    this->headers.fmt[3] = '\0';
    this->headers.subchunk1Size = 0;
    this->headers.audioFormat = 0;
    this->headers.numOfChan = 0;
    this->headers.samplingrate = 0;
    this->headers.bytesPerSec = 0;
    this->headers.blockAlign = 0;
    this->headers.bitsPerSample = 0;
    /* "data" sub-chunk */
    this->headers.subchunk2ID[0] = '\0';
    this->headers.subchunk2ID[1] = '\0';
    this->headers.subchunk2ID[2] = '\0';
    this->headers.subchunk2ID[3] = '\0';
    this->headers.subchunk2Size = 0;
}

void Wav::initializeHeaders(uint32_t samplingRate, uint16_t bitsPerSample, uint16_t numOfChan,
                            size_t numSamples)
{
    this->headers.RIFF[0] = 'R';
    this->headers.RIFF[1] = 'I';
    this->headers.RIFF[2] = 'F';
    this->headers.RIFF[3] = 'F';
    this->headers.WAVE[0] = 'W';
    this->headers.WAVE[1] = 'A';
    this->headers.WAVE[2] = 'V';
    this->headers.WAVE[3] = 'E';
    this->headers.fmt[0] = 'f';
    this->headers.fmt[1] = 'm';
    this->headers.fmt[2] = 't';
    this->headers.fmt[3] = ' ';
    this->headers.subchunk1Size = 16; // PCM
    this->headers.audioFormat = 1;    // PCM
    this->headers.numOfChan = numOfChan;
    this->headers.samplingrate = samplingRate;
    this->headers.bitsPerSample = bitsPerSample;
    this->headers.bytesPerSec = samplingRate * numOfChan * (bitsPerSample / 8);
    this->headers.blockAlign = numOfChan * (bitsPerSample / 8);
    this->headers.subchunk2ID[0] = 'd';
    this->headers.subchunk2ID[1] = 'a';
    this->headers.subchunk2ID[2] = 't';
    this->headers.subchunk2ID[3] = 'a';
    this->headers.subchunk2Size = numSamples * numOfChan * (bitsPerSample / 8);
    this->headers.chunkSize = 36 + this->headers.subchunk2Size;
}