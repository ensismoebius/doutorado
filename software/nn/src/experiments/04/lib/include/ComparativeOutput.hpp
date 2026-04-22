#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "ComparativeConfig.hpp"
#include "ResultRow.hpp"

namespace comparative_autoencoder_experiment
{

void write_rows_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_publication_table(const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_summary_json(const std::filesystem::path& path,
    const ComparativeConfig& cfg,
    std::size_t cfg_hash,
    const std::vector<ResultRow>& rows);

void validate_repeat_determinism(const ComparativeConfig& cfg, const std::vector<ResultRow>& rows);

} // namespace comparative_autoencoder_experiment
