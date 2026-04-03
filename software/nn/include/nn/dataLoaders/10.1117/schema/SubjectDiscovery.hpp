/**
 * @file include/nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp
 * @brief Subjectdiscovery.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_DATALOADERS_10_1117_SUBJECTDISCOVERY_HPP
#define NN_DATALOADERS_10_1117_SUBJECTDISCOVERY_HPP

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

#endif // NN_DATALOADERS_10_1117_SUBJECTDISCOVERY_HPP