#include <filesystem> // Include for path manipulation
#include <iostream>
#include <string>

#include "core/dataLoaders/AudioLoader.h"
#include "core/dataLoaders/EEGLoader.h"
#include "core/dataLoaders/MatFile.h"
#include "core/wave/Wav.h"

using std::cout;

using namespace nn::dataLoaders;

static void init()
{
    // Initialization code goes here
}

static void perform(const std::string& basePath)
{
    for (const auto& entry : std::filesystem::directory_iterator(basePath))
    {
        if (entry.is_directory())
        {
            std::string subjectPath = entry.path().string();
            std::string subjectName = entry.path().filename().string();
            std::string audioFilePath = subjectPath + "/" + subjectName + "_Audio.mat";
            std::string eegFilePath = subjectPath + "/" + subjectName + "_EEG.mat";

            if (std::filesystem::exists(audioFilePath) && std::filesystem::exists(eegFilePath))
            {
                cout << "Processing subject: " << subjectName << '\n';

                auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);

                // The eegIndex from the audio file is 1-based, so we subtract 1 for 0-based
                // indexing
                long eegRowIndex = static_cast<long>(eegIndex) - 1;

                auto [eegSamplesMatrix, eegInfo] = loadEEGFromMat(eegFilePath, eegRowIndex);

                // Split EEG data into channels
                constexpr int numChannels = 6;
                std::vector<Eigen::VectorXf> eegChannels(numChannels);

                for (int j = 0; j < numChannels; ++j)
                {
                    eegChannels[j] = eegSamplesMatrix.row(j);
                }

                // Now you have:
                // - audioSamples (Eigen::VectorXf)
                // - eegChannels (std::vector of 6 Eigen::VectorXf)
                // You can proceed with further processing...

                cout << "  - Loaded Audio Sample linked to EEG Sample " << eegRowIndex << '\n';

                // Example: Write audio to a WAV file
                std::vector<float> audioSamplesVec(audioSamples.data(),
                                                   audioSamples.data() + audioSamples.size());
                std::vector<double> audioSamplesDoubleVec(audioSamplesVec.begin(),
                                                          audioSamplesVec.end());

                Wav w(44100, 16, 1, audioSamplesDoubleVec.data(), audioSamplesDoubleVec.size());
                std::filesystem::path subjectDirPath(subjectPath);
                std::string outputFilename = subjectName + "_AudioSample.wav";
                std::string outputWavPath = (subjectDirPath / outputFilename).string();
                w.write(outputWavPath);

                for (int j = 0; j < numChannels; ++j)
                {
                    std::vector<double> eegChannelDoubleVec(
                        eegChannels[j].data(), eegChannels[j].data() + eegChannels[j].size());
                    Wav eegWav(1024, 16, 1, eegChannelDoubleVec.data(), eegChannelDoubleVec.size());
                    std::string eegOutputFilename =
                        subjectName + "_EEG_Channel" + std::to_string(j + 1) + ".wav";
                    std::string eegOutputWavPath =
                        (std::filesystem::path(subjectPath) / eegOutputFilename).string();
                    eegWav.write(eegOutputWavPath);
                    cout << "  - Wrote EEG Channel " << (j + 1) << " to " << eegOutputWavPath
                         << '\n';
                }
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