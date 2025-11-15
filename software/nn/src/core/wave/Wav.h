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

    /**
     * @brief WAV file header structure
     *  This structure defines the layout of the WAV file header, which contains metadata
     *  about the audio file, such as format, number of channels, sample rate, and bit depth.
     */
    struct WaveHeader
    {
        /* RIFF Chunk Descriptor */
        std::array<std::uint8_t, 4> riffChunkId; // "RIFF" Header, contains the letters "RIFF"
        std::uint32_t riffChunkSize;             // Size of the overall file - 8 bytes
        std::array<std::uint8_t, 4> waveFormat;  // "WAVE" Format, contains the letters "WAVE"

        /* "fmt" sub-chunk */
        std::array<std::uint8_t, 4> fmtSubchunkId; // "fmt" header, contains the letters "fmt "
        std::uint32_t fmtSubchunkSize;             // Size of the fmt chunk (16 for PCM)
        std::uint16_t audioFormat;      // Audio format 1=PCM,6=mulaw,7=alaw,257=IBM Mu-Law, 258=IBM
                                        // A-Law, 259=ADPCM
        std::uint16_t numberOfChannels; // Number of channels 1=Mono 2=Stereo
        std::uint32_t sampleRate;       // Sampling Frequency in Hz
        std::uint32_t byteRate;         // bytes per second
        std::uint16_t blockAlign;       // 2=16-bit mono, 4=16-bit stereo
        std::uint16_t bitsPerSample;    // Number of bits per sample

        /* "data" sub-chunk */
        std::array<std::uint8_t, 4> dataSubchunkId; // "data" header, contains the letters "data"
        std::uint32_t dataSubchunkSize;             // Size of the data section in bytes
    } header;
#pragma pack(pop)

    // Wave format keys, used to identify audio format based on resolution and channels
    // 8-bit mono, 8-bit stereo, 16-bit mono, 16-bit stereo
    // The format key is created by multiplying the wave resolution by 10 and adding
    // the number of channels.
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
