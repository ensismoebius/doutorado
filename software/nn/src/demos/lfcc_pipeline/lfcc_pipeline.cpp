/**
 * @file lfcc_pipeline.cpp
 * @brief Demo driver for LFCC feature extraction over a dataset directory.
 *
 * This executable walks a dataset folder structure, locates `*_Audio.mat`/`*_EEG.mat`
 * pairs per subject, and delegates LFCC extraction to `lfcc_pipeline_utils`.
 */

#include <filesystem>
#include <string>

#include "demos/lfcc_pipeline/lfcc_pipeline_utils.h"
#include "nn/wave/audioTypes.h"

using std::string;

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
            string subjectPath = entry.path().string();
            string subjectName = entry.path().filename().string();
            string audioFilePath = subjectPath;
            audioFilePath += "/";
            audioFilePath += subjectName;
            audioFilePath += "_Audio.mat";
            string eegFilePath = subjectPath;
            eegFilePath += "/";
            eegFilePath += subjectName;
            eegFilePath += "_EEG.mat";

            if (std::filesystem::exists(audioFilePath) && std::filesystem::exists(eegFilePath))
            {
                const SubjectInfo subject = {.path = subjectPath,
                                             .name = subjectName,
                                             .audio_file_path = audioFilePath,
                                             .eeg_file_path = eegFilePath};
                processSubject(subject);
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