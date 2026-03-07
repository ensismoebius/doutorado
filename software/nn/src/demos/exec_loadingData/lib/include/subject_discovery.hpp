#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct SubjectFiles
{
    int subject_id = 0;
    std::string subject_name;
    std::string eeg_mat_path;
    std::string audio_mat_path;
    std::size_t eeg_rows = 0;
    std::size_t audio_rows = 0;
};

auto discoverSubjects(const std::string& root_dir, const std::string& subject_regex_pattern)
    -> std::vector<SubjectFiles>;

