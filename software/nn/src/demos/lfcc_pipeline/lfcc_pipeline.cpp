/**
 * @file lfcc_pipeline.cpp
 * @brief Demo driver for LFCC feature extraction over a dataset directory.
 *
 * This executable walks a dataset folder structure, locates `*_Audio.mat`/`*_EEG.mat`
 * pairs per subject, and delegates LFCC extraction to `lfcc_pipeline_utils`.
 */

#include <filesystem>
#include <string>

#include "wave/audioTypes.h"
#include "wave/lfcc_pipeline_utils.h"

using std::string;

static void init()
{
    // Initialization code goes here
}

static void perform(const std::string& base_path)
{
    for (const auto& entry : std::filesystem::directory_iterator(base_path))
    {
        if (entry.is_directory())
        {
            string subject_path = entry.path().string();
            string subject_name = entry.path().filename().string();
            string audio_file_path = subject_path;
            audio_file_path += "/";
            audio_file_path += subject_name;
            audio_file_path += "_Audio.mat";
            string eeg_file_path = subject_path;
            eeg_file_path += "/";
            eeg_file_path += subject_name;
            eeg_file_path += "_EEG.mat";

            if (std::filesystem::exists(audio_file_path) && std::filesystem::exists(eeg_file_path))
            {
                const SubjectInfo subject = {.path = subject_path,
                    .name = subject_name,
                    .audio_file_path = audio_file_path,
                    .eeg_file_path = eeg_file_path};
                process_subject(subject);
            }
        }
    }
}

auto main(int argc, char** argv) -> int
{
    init();

    std::string base_path =
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/";

    perform(base_path);

    return 0;
}