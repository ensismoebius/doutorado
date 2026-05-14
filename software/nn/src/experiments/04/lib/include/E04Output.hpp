#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "E04Config.hpp"
#include "E04EpochHistory.hpp"
#include "E04ResultRow.hpp"

namespace e04
{

void write_rows_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_publication_table(const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_summary_json(const std::filesystem::path& path,
    const E04Config& cfg,
    std::size_t cfg_hash,
    const std::vector<ResultRow>& rows);

void write_latex_exports(const std::filesystem::path& dir,
    const std::string& run_tag,
    const E04Config& cfg,
    const std::vector<ResultRow>& rows);

void validate_repeat_determinism(const E04Config& cfg, const std::vector<ResultRow>& rows);

void write_pgfplots_summary_dat(
    const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_pgfplots_sweep_dat(
    const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_epoch_history_dat(const std::filesystem::path& path,
    const std::string& model,
    const std::string& encoding,
    const std::string& architecture,
    float v_th,
    float alpha,
    int run_id,
    const EpochHistory& history);

void write_batch_convergence_dat(const std::filesystem::path& path,
    const std::string& model,
    const std::string& encoding,
    const std::string& architecture,
    float v_th,
    float alpha,
    int run_id,
    const EpochHistory& history);

} // namespace e04

