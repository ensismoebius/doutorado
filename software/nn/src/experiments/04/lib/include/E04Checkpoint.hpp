#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "E04EpochHistory.hpp"
#include "E04ResultRow.hpp"

namespace e04
{

struct CheckpointKey
{
    std::string run_tag;
    std::string backend;
    std::string dataset;
    std::string model;
    std::string encoding;
    std::string architecture;
    float v_th{0.0f};
    float alpha{0.0f};
    int run_id{1};
};

auto checkpoint_path(const std::filesystem::path& chk_dir, const CheckpointKey& key)
    -> std::filesystem::path;

auto checkpoint_is_valid(const std::filesystem::path& path, std::size_t expected_hash) -> bool;

auto checkpoint_load(const std::filesystem::path& path) -> ResultRow;

void checkpoint_save(const std::filesystem::path& path,
                     const ResultRow& row,
                     const EpochHistory& history,
                     std::size_t config_hash);

} // namespace e04
