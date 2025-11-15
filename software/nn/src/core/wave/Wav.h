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
#ifndef SRC_LIB_WAV_H_
#define SRC_LIB_WAV_H_

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class Wav
{
   private:
    // signal properties
#pragma pack(push, 1)
    struct WaveHeader
    {
        /* RIFF Chunk Descriptor */
        std::array<std::uint8_t, 4> riffChunkId;
        std::uint32_t riffChunkSize;
        std::array<std::uint8_t, 4> waveFormat;

        /* "fmt" sub-chunk */
        std::array<std::uint8_t, 4> fmtSubchunkId;
        std::uint32_t fmtSubchunkSize;
        std::uint16_t audioFormat;
        std::uint16_t numberOfChannels;
        std::uint32_t sampleRate;
        std::uint32_t byteRate;
        std::uint16_t blockAlign;
        std::uint16_t bitsPerSample;

        /* "data" sub-chunk */
        std::array<std::uint8_t, 4> dataSubchunkId;
        std::uint32_t dataSubchunkSize;
    } header;
#pragma pack(pop)

    enum WaveFormatKey : std::uint8_t
    {
        Format8BitMono = 81,
        Format8BitStereo = 82,
        Format16BitMono = 161,
        Format16BitStereo = 162
    };

    // another signal properties
    size_t amountOfData;
    int waveResolution;

    // signal data
    std::vector<double> data;
    std::vector<double> dataLeft;
    std::vector<double> dataRight;

    // path of file containing the signal
    std::string path;

    // callback function (applied on data)
    void (*callbackFunction)(        //
        std::vector<double>& signal, //
        size_t& signalLength,        //
        uint32_t samplingRate,       //
        std::string path             //
    );

   public:
    Wav(); // Default constructor
    Wav(uint32_t samplingRate, uint16_t bitsPerSample, uint16_t numOfChan, const double* audioData,
        size_t audioDataSize);
    /**
     * Process the data
     */
    void process();

    /**
     * Read the wav file
     * @param path
     */
    void read(const std::string& _path);

    /**
     * Write a wav file
     * @param path
     */
    void write(const std::string& _path);

    /**
     * Write a wav file from a vector (mono)
     * @param path
     * @param data
     * @param samplingRate
     */
    void write(const std::string& _path, const std::vector<float>& data, int samplingRate);

    /**
     * Write a wav file from a bi-dimensional vector (multi-channel)
     * @param path
     * @param data
     * @param samplingRate
     */
    void write(const std::string& _path, const std::vector<std::vector<float>>& data,
               int samplingRate);

    /**
     * Returns the path of file containing the signal
     * @return path
     */
    [[nodiscard]] auto getPath() const -> std::string;

    /**
     * Returns the raw wav data (monolitic wav)
     * @return data
     */
    [[nodiscard]] auto getData() const -> std::vector<double>;

    /**
     * Returns the raw wav data (left channel wav)
     * @return data
     */
    [[nodiscard]] auto getDataLeft() const -> std::vector<double>;

    /**
     * Returns the raw wav data (right channel wav)
     * @return data
     */
    [[nodiscard]] auto getDataRight() const -> std::vector<double>;

    /**
     * The function witch will manipulate the wav data
     * @param callbackFunction
     */
    void setCallbackFunction(            //
        void (*_callbackFunction)(       //
            std::vector<double>& signal, //
            size_t& signalLength,        //
            uint32_t samplingRate,       //
            std::string path             //
            )                            // callback function
    );

   private:
    /**
     * @brief Converts two 8-bit unsigned chars (LSB and MSB) to a 16-bit signed short.
     *
     * This function is used to reconstruct a 16-bit audio sample from the two 8-bit bytes
     * read from a WAV file. It assumes little-endian byte order, where the
     * least significant byte (LSB) is followed by the most significant byte (MSB).
     *
     * @param lsb The least significant byte of the 16-bit sample.
     * @param msb The most significant byte of the 16-bit sample.
     * @return The reconstructed 16-bit signed sample as a short.
     */
    static auto combine8BitTo16Bit(unsigned char lsb, unsigned char msb) -> short;

    /**
     * @brief Converts a 16-bit signed short into two 8-bit unsigned chars (LSB and MSB).
     *
     * This function is used to split a 16-bit audio sample into two 8-bit bytes
     * for writing to a WAV file in little-endian format.
     *
     * @param sample The 16-bit signed sample to convert.
     * @param lsb Pointer to an unsigned char to store the least significant byte.
     * @param msb Pointer to an unsigned char to store the most significant byte.
     */
    static void split16BitTo8Bit(short sample, unsigned char* lsb, unsigned char* msb);

    void initializeHeaders(uint32_t samplingRate, uint16_t bitsPerSample, uint16_t numberOfChannels,
                           size_t numSamples);
    void readWaveData(std::ifstream& ifs);
    void readWaveHeaders(std::ifstream& ifs);
    inline void write8BitMono(std::ofstream& ofs);
    inline void write8BitStereo(std::ofstream& ofs);
    inline void write16BitMono(std::ofstream& ofs);
    inline void write16BitStereo(std::ofstream& ofs);
    inline void read8BitMono(std::ifstream& ifs);
    inline void read8BitStereo(std::ifstream& ifs);
    inline void read16BitMono(std::ifstream& ifs);
    inline void read16BitStereo(std::ifstream& ifs);
    void clearVectors();
    void resetMetaData();
};
#endif /* SRC_LIB_WAV_H_ */
