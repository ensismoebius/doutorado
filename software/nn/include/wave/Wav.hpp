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

/**
 * @file Wav.hpp
 * @brief Minimal WAV (PCM) reader/writer used by audio experiments.
 *
 * This is a self-contained utility for loading and saving WAV files and exposing
 * the decoded samples as `std::vector<double>`.
 *
 * Notes:
 * - Intended for offline preprocessing/feature extraction rather than real-time audio.
 * - Current implementation focuses on PCM formats.
 */
#ifndef SRC_LIB_WAV_H_
#define SRC_LIB_WAV_H_

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

/**
 * @brief WAV file class for reading, writing, and processing audio data.
 *  This class provides functionality to read and write WAV files,
 *  manipulate audio data through callback functions, and manage
 *  audio metadata such as sample rate, bit depth, and number of channels.
 *  For now it just supports PCM format.
 */
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
    /**
     * @brief The number of audio samples in the data vectors.
     */
    size_t amountOfData;

    /**
     * @brief The resolution of the wave file in bits per sample (e.g., 8 or 16).
     */
    int waveResolution;

    // signal data
    /**
     * @brief Vector to store mono audio data as doubles.
     */
    std::vector<double> data;

    /**
     * @brief Vector to store the left channel of stereo audio data as doubles.
     */
    std::vector<double> dataLeft;

    /**
     * @brief Vector to store the right channel of stereo audio data as doubles.
     */
    std::vector<double> dataRight;

    /**
     * @brief The file path for the WAV file being read or written.
     */
    std::string path;

    /**
     * @brief Pointer to the callback function for processing audio data.
     *
     * This function pointer allows users to set a custom function that will
     * manipulate the audio data stored in the Wav object. The callback function
     * should match the specified signature, taking a vector of doubles (the audio
     * signal), a reference to the signal length, the sampling rate, and the file path.
     */
    void (*callbackFunction)(std::vector<double>& signal,
        size_t& signalLength,
        uint32_t samplingRate,
        const std::string& path) = nullptr;

    /**
     * @brief Helper function to read binary data from a stream.
     * @tparam T The type of the data to read.
     * @param is The input stream.
     * @param value The variable to store the read data.
     * @return Reference to the input stream.
     */
    template <typename T>
    static std::istream& read_binary(std::istream& is, T& value)
    {
        return is.read(reinterpret_cast<char*>(std::addressof(value)), sizeof(T));
    }

    /**
     * @brief Helper function to write binary data to a stream.
     * @tparam T The type of the data to write.
     * @param os The output stream.
     * @param value The data to write.
     * @return Reference to the output stream.
     */
    template <typename T>
    static std::ostream& write_binary(std::ostream& os, const T& value)
    {
        return os.write(reinterpret_cast<const char*>(std::addressof(value)), sizeof(T));
    }

   public:
    /**
     * @brief Construct a new Wav object
     *
     */
    Wav();

    /**
     * @brief Construct a new Wav object
     *
     * @param samplingRate
     * @param bitsPerSample
     * @param numOfChan
     * @param audioData
     * @param audioDataSize
     */
    Wav(uint32_t samplingRate,
        uint16_t bitsPerSample,
        uint16_t numOfChan,
        const double* audioData,
        size_t audioDataSize);

    /**
     * @brief Process the audio data using the set callback function.
     */
    void process();

    /**
     * @brief Read a wav file, extracting its audio data and metadata.
     * @param path The file path of the wav file to read.
     */
    void read(const std::string& _path); // flawfinder: ignore

    /**
     * @brief Write a wav file, including its audio data and metadata.
     * @param path The file path to write the wav file to.
     */
    void write(const std::string& _path);

    /**
     * Write a wav file from a vector (mono)
     * @param path The file path to write the wav file to.
     * @param data The audio data to write.
     * @param samplingRate The sampling rate of the audio data.
     */
    void write(const std::string& _path, const std::vector<float>& data, int samplingRate);

    /**
     * Write a wav file from a bi-dimensional vector (multi-channel)
     * @param path The file path to write the wav file to.
     * @param data The audio data to write.
     * @param samplingRate The sampling rate of the audio data.
     */
    void write(
        const std::string& _path, const std::vector<std::vector<float>>& data, int samplingRate);

    /**
     * @brief Returns the path of file containing the signal
     * @return path The file path of the wav file.
     */
    [[nodiscard]] auto get_path() const -> std::string;

    /**
     * @brief Returns the raw wav data (monolithic wav)
     * @return data The raw wav data as a vector of doubles.
     */
    [[nodiscard]] auto get_data() const -> const std::vector<double>&;

    /**
     * @brief Returns the raw wav data (left channel wav)
     * @return data The raw wav data for the left channel as a vector of doubles.
     */
    [[nodiscard]] auto get_data_left() const -> const std::vector<double>&;

    /**
     * @brief Returns the raw wav data (right channel wav)
     * @return data The raw wav data for the right channel as a vector of doubles.
     */
    [[nodiscard]] auto get_data_right() const -> const std::vector<double>&;

    /**
     * The function which will manipulate the wav data
     * @param callbackFunction The callback function to set
     */
    void set_callback_function(          //
        void (*_callbackFunction)(       //
            std::vector<double>& signal, //
            size_t& signalLength,        //
            uint32_t samplingRate,       //
            const std::string& path      //
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
    static auto combine_8bit_to_16bit(unsigned char lsb, unsigned char msb) -> short;

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
    static void split_16bit_to_8bit(short sample, unsigned char* lsb, unsigned char* msb);

    /**
     * @brief Initializes the WAV file header with the specified audio properties.
     * @param samplingRate The sample rate in Hz.
     * @param bitsPerSample The number of bits per sample.
     * @param numberOfChannels The number of audio channels (1 for mono, 2 for stereo).
     * @param numSamples The total number of samples.
     */
    void initialize_headers(uint32_t samplingRate,
        uint16_t bitsPerSample,
        uint16_t numberOfChannels,
        size_t numSamples);
    /**
     * @brief Reads the audio data from the WAV file based on the format.
     * @param ifs The input file stream to read from.
     */
    void read_wave_data(std::ifstream& ifs);

    /**
     * @brief Reads the WAV file headers from the input stream.
     * @param ifs The input file stream to read from.
     */
    void read_wave_headers(std::ifstream& ifs);

    /**
     * @brief Writes 8-bit mono audio data to the output stream.
     * @param ofs The output file stream to write to.
     */
    inline void write8BitMono(std::ofstream& ofs);

    /**
     * @brief Writes 8-bit stereo audio data to the output stream.
     * @param ofs The output file stream to write to.
     */
    inline void write8BitStereo(std::ofstream& ofs);

    /**
     * @brief Writes 16-bit mono audio data to the output stream.
     * @param ofs The output file stream to write to.
     */
    inline void write16BitMono(std::ofstream& ofs);

    /**
     * @brief Writes 16-bit stereo audio data to the output stream.
     * @param ofs The output file stream to write to.
     */
    inline void write16BitStereo(std::ofstream& ofs);

    /**
     * @brief Reads 8-bit mono audio data from the input stream.
     * @param ifs The input file stream to read from.
     */
    inline void read8BitMono(std::ifstream& ifs);

    /**
     * @brief Reads 8-bit stereo audio data from the input stream.
     * @param ifs The input file stream to read from.
     */
    inline void read8BitStereo(std::ifstream& ifs);

    /**
     * @brief Reads 16-bit mono audio data from the input stream.
     * @param ifs The input file stream to read from.
     */
    inline void read16BitMono(std::ifstream& ifs);

    /**
     * @brief Reads 16-bit stereo audio data from the input stream.
     * @param ifs The input file stream to read from.
     */
    inline void read16BitStereo(std::ifstream& ifs);

    /**
     * @brief Clears all audio data vectors (mono, left, and right channels).
     */
    void clear_vectors();

    /**
     * @brief Resets all metadata of the Wav object to their default states.
     */
    void reset_metadata();
};
#endif /* SRC_LIB_WAV_H_ */
