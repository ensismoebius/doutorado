// TODO: Refactor this experiment into an auto-encoder training experiment which uses
// the AudioLoader and EEGLoader to load data, and trains an traditional auto-encoder model
// as well as a spiking auto-encoder model on the audio and EEG data respectively.
// Then compare, using paraconsistent features enginering techniques, to assesss the
// performance of both models in terms of the quality of the generated features vectors.
// The input data must be downsampled to 

#include <filesystem> // Include for path manipulation
#include <iostream>
#include <map> // Added for std::map
#include <string>
#include <vector> // Added for std::vector

#include "core/dataLoaders/10.1117/AudioData.h"
#include "core/dataLoaders/10.1117/AudioLoader.h"
#include "core/dataLoaders/10.1117/EEGData.h"
#include "core/dataLoaders/10.1117/EEGLoader.h"
#include "core/wave/Wav.h"

using std::cout;
using std::string;

using namespace nn::dataLoaders;

static void init()
{
    // Initialization code goes here
}

static auto loadAndProcessAudio(const std::string& audioFilePath) -> AudioData
{
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);
    return {
        .audioSamples = audioSamples,               // Eigen::VectorXf
        .audioStimulus = audioStimulus,             // int
        .eegIndex = static_cast<long>(eegIndex) - 1 // Convert to 0-based index
    };
}

static auto loadAndProcessEEG(const std::string& eegFilePath, long eegRowIndex) -> EEGData
{
    // The number of channels is fixed at 6 for this dataset
    constexpr int numChannels = 6;

    auto [eegSamplesMatrix, eegInfo] = loadEEGFromMat(eegFilePath, eegRowIndex);

    // Split eegSamplesMatrix into individual channel vectors
    std::vector<Eigen::VectorXf> eegChannels(numChannels);
    for (int j = 0; j < numChannels; ++j)
    {
        eegChannels[j] = eegSamplesMatrix.row(j);
    }

    return {
        .eegSamplesMatrix = eegSamplesMatrix, // Eigen::MatrixXf
        .eegInfo = eegInfo,                   // std::array<int, 3>
        .eegChannels = eegChannels            // std::vector<Eigen::VectorXf>
    };
}

static void writeAudioToWav(const Eigen::VectorXf& audioSamples, int audioStimulus,
                            const std::string& subjectPath, const std::string& subjectName)
{
    // Get stimulus name for audio
    std::string audioStimulusName = "Unknown";
    if (ESTIMULUS_NAMES.contains(audioStimulus))
    {
        audioStimulusName = ESTIMULUS_NAMES.at(audioStimulus);
    }

    // Cast Eigen::VectorXf to std::vector<double>
    std::vector<double> audioSamplesVec(audioSamples.data(),
                                        audioSamples.data() + audioSamples.size());

    // Construct output WAV file path
    std::filesystem::path subjectDirPath(subjectPath);
    std::string outputFilename = subjectName + "_AudioSample_" + audioStimulusName + ".wav";
    std::string outputWavPath = (subjectDirPath / outputFilename).string();

    // Assuming audio sample rate is 44100 Hz, 16-bit, 1 channel for this dataset
    Wav w(44100, 16, 1, audioSamplesVec.data(), audioSamplesVec.size());
    w.write(outputWavPath);

    cout << "  - Wrote Audio Sample to " << outputWavPath << '\n';
}

static void writeEEGToWav(const std::vector<Eigen::VectorXf>& eegChannels, int eegStimulus,
                          const std::string& subjectPath, const std::string& subjectName)
{
    // Get stimulus name for EEG
    std::string eegStimulusName = "Unknown";
    if (ESTIMULUS_NAMES.contains(eegStimulus))
    {
        eegStimulusName = ESTIMULUS_NAMES.at(eegStimulus);
    }

    // Write each EEG channel to a separate WAV file
    for (int j = 0; j < eegChannels.size(); ++j)
    {
        // Convert Eigen::VectorXf to std::vector<double>
        std::vector<double> eegChannelDoubleVec(eegChannels[j].data(),
                                                eegChannels[j].data() + eegChannels[j].size());

        // Construct output WAV file path
        std::string eegOutputFilename =
            subjectName + "_EEG_Channel_" + EEG_CHANNELS_NAMES[j] + "_" + eegStimulusName + ".wav";
        std::string eegOutputWavPath =
            (std::filesystem::path(subjectPath) / eegOutputFilename).string();

        // Assuming EEG sample rate is 1024 Hz, 16-bit, 1 channel
        Wav eegWav(1024, 16, 1, eegChannelDoubleVec.data(), eegChannelDoubleVec.size());
        eegWav.write(eegOutputWavPath);

        cout << "  - Wrote EEG Channel " << EEG_CHANNELS_NAMES[j] + " to " + eegOutputWavPath
             << '\n';
    }
}

static void processSubject(const std::string& subjectPath, const std::string& subjectName,
                           const std::string& audioFilePath, const std::string& eegFilePath)
{
    cout << "Processing subject: " << subjectName << '\n';

    AudioData audioData = loadAndProcessAudio(audioFilePath);
    EEGData eegData = loadAndProcessEEG(eegFilePath, audioData.eegIndex);

    writeAudioToWav(audioData.audioSamples, audioData.audioStimulus, subjectPath, subjectName);

    // Get EEG stimulus from eegInfo, for this dataset index 1 corresponds to stimulus
    int eegStimulus = eegData.eegInfo[1];

    writeEEGToWav(eegData.eegChannels, eegStimulus, subjectPath, subjectName);
}

static void perform(const std::string& basePath)
{
    for (const auto& entry : std::filesystem::directory_iterator(basePath))
    {
        if (entry.is_directory())
        {
            string subjectPath = entry.path().string();
            string subjectName = entry.path().filename().string();
            string audioFilePath = subjectPath + "/" + subjectName + "_Audio.mat";
            string eegFilePath = subjectPath + "/" + subjectName + "_EEG.mat";

            if (std::filesystem::exists(audioFilePath) && std::filesystem::exists(eegFilePath))
            {
                processSubject(subjectPath, subjectName, audioFilePath, eegFilePath);
            }
        }
    }
}

auto main(int argc, char** argv) -> int
{
    init();

    std::string basePath =
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/";

    perform(basePath);

    return 0;
}