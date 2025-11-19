#ifndef EXPERIMENT01_UTILS_HPP
#define EXPERIMENT01_UTILS_HPP

#include <string>

struct AudioProcessingParameters
{
    int target_sampling_rate;
    double preemphasis_coefficient;
    double frame_duration_ms;
    double frame_shift_ms;
    int number_of_filters;
    int number_of_cepstrals;
    int delta_window_span;
};

struct SubjectInfo
{
    std::string path;
    std::string name;
    std::string audio_file_path;
    std::string eeg_file_path;
};

void processSubject(const SubjectInfo& subject);

#endif // EXPERIMENT01_UTILS_HPP
