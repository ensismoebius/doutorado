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

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class Wav
{
   private:
    // signal properties
#pragma pack(push, 1)
    struct
    {
        /* RIFF Chunk Descriptor */
        std::uint8_t RIFF[4];    // RIFF Header Magic header
        std::uint32_t chunkSize; // RIFF Chunk Size
        std::uint8_t WAVE[4];    // WAVE Header

        /* "fmt" sub-chunk */
        std::uint8_t fmt[4];         // FMT header
        std::uint32_t subchunk1Size; // Size of the fmt chunk
        std::uint16_t audioFormat;   // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM
                                     // A-Law, 259=ADPCM
        std::uint16_t numOfChan;     // Number of channels 1=Mono 2=Sterio
        std::uint32_t samplingrate;  // Sampling Frequency in Hz
        std::uint32_t bytesPerSec;   // bytes per second
        std::uint16_t blockAlign;    // 2=16-bit mono, 4=16-bit stereo
        std::uint16_t bitsPerSample; // Number of bits per sample

        /* "data" sub-chunk */
        std::uint8_t subchunk2ID[4]; // "data"  string
        std::uint32_t subchunk2Size; // Sampled data length
    } headers;
#pragma pack(pop)

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
    auto convert2of8to1of16(unsigned char lsb, unsigned char msb) -> short;
    void convert1of16to2of8(short result, unsigned char* lsb, unsigned char* msb);
    void readWaveData(std::ifstream& ifs);
    void readWaveHeaders(std::ifstream& ifs);
    inline void write8Res1Channel(std::ofstream& ofs);
    inline void write8Res2Channel(std::ofstream& ofs);
    inline void write16Res1Channel(std::ofstream& ofs);
    inline void write16Res2Channel(std::ofstream& ofs);
    inline void read8Res1Channel(std::ifstream& ifs);
    inline void read8Res2Channel(std::ifstream& ifs);
    inline void read16Res1Channel(std::ifstream& ifs);
    inline void read16Res2Channel(std::ifstream& ifs);
    void clearVectors();
    void resetMetaData();
};
#endif /* SRC_LIB_WAV_H_ */
