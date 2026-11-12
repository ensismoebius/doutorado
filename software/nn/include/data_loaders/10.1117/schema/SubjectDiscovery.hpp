/**
 * @file include/data_loaders/10.1117/schema/SubjectDiscovery.hpp
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
#include <map>
#include <string>
#include <vector>

struct SubjectFiles
{
    int subject_id = 0;
    std::string subject_name;
    std::string eeg_path;
    std::string audio_path;
    std::size_t eeg_rows = 0;
    std::size_t audio_rows = 0;
};

auto discoverSubjects(const std::string& root_dir, const std::string& subject_regex_pattern)
    -> std::vector<SubjectFiles>;

// Stimulus id -> name from a .sqlite dataset's `stimulus` table, the table that
// trial.stimulus_id references. Throws if root_dir is not a .sqlite file, or the table
// is unreadable or empty, or a name is NULL/empty.
auto discoverStimulusNames(const std::string& root_dir) -> std::map<int, std::string>;

#endif // NN_DATALOADERS_10_1117_SUBJECTDISCOVERY_HPP