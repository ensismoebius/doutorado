/**
 * @file src/experiments/autoencoderRunner/lib/include/ProfileJsonParser.hpp
 * @brief Hand-rolled JSON parsing helpers for profile files (extracted from
 *        ProfileLoader.cpp).
 *
 * Everything here is internal plumbing for ProfileLoader.cpp — not part of
 * this library's public API — hence `namespace autoencoderRunner::detail`.
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "cli.hpp"

namespace autoencoderRunner::detail
{

// ── Profile-domain token mapping (used by ProfileLoader.cpp's apply_* helpers) ──

auto map_dataset_type(const std::string& s, Config& cfg) -> bool;
auto map_autoencoder_type(const std::string& s, Config& cfg) -> bool;

// ── Raw file / key lookup ────────────────────────────────────────────────────

auto read_file(const std::filesystem::path& path, std::string& out) -> bool;
auto find_key(const std::string& text, const std::string& key) -> std::size_t;

// value_start()/parse_token() are declared (not anonymous-namespaced) only
// because parse_number<T> below is a template and must see them from here;
// nothing outside this file calls them directly.
auto value_start(const std::string& text, const std::string& key, std::size_t& out_pos) -> bool;
auto parse_token(const std::string& text, std::size_t start, std::string& out) -> bool;

// ── Typed value extraction ───────────────────────────────────────────────────

template <typename T>
auto parse_number(const std::string& text, const std::string& key, T& out) -> bool
{
    std::size_t pos = 0;
    if (!value_start(text, key, pos)) return false;
    std::string token;
    if (!parse_token(text, pos, token)) return false;
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str()) return false;
    out = static_cast<T>(value);
    return true;
}

auto parse_bool(const std::string& text, const std::string& key, bool& out) -> bool;
auto parse_string(const std::string& text, const std::string& key, std::string& out) -> bool;
auto parse_array_numbers(const std::string& text, const std::string& key, std::vector<double>& out)
    -> bool;
auto parse_array_ints(const std::string& text, const std::string& key, std::vector<int>& out)
    -> bool;
auto parse_array_strings(
    const std::string& text, const std::string& key, std::vector<std::string>& out) -> bool;
auto parse_object(const std::string& text, const std::string& key, std::string& out) -> bool;

// ── Unknown-key validation ───────────────────────────────────────────────────

auto validate_known_profile_keys(const std::string& text, std::string& out_error) -> bool;

} // namespace autoencoderRunner::detail
