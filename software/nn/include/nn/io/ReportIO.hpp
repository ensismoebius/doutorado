/**
 * @file ReportIO.hpp
 * @brief Shared helpers for report naming and file emission.
 */

#ifndef NN_IO_REPORTIO_HPP
#define NN_IO_REPORTIO_HPP

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace nn::io
{

inline auto sanitize_stem(std::string stem) -> std::string
{
    if (stem.empty()) stem = "profile";
    for (char& c : stem)
    {
        const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
        if (!ok) c = '_';
    }
    return stem;
}

inline auto timestamp_now_compact_local() -> std::string
{
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

inline auto write_text_file(
    const std::filesystem::path& path, const std::string& content, std::string* out_error = nullptr)
    -> bool
{
    std::ofstream out(path);
    if (!out)
    {
        if (out_error)
        {
            *out_error = "failed to open output file: " + path.string();
        }
        return false;
    }
    out << content;
    if (!out.good())
    {
        if (out_error)
        {
            *out_error = "failed while writing output file: " + path.string();
        }
        return false;
    }
    return true;
}

inline auto write_json_file(const std::filesystem::path& path,
    const nlohmann::json& payload,
    int indent = 2,
    std::string* out_error = nullptr) -> bool
{
    return write_text_file(path, payload.dump(indent) + "\n", out_error);
}

} // namespace nn::io

#endif // NN_IO_REPORTIO_HPP