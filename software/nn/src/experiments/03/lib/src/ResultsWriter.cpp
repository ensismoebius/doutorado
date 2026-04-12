/**
 * @file src/experiments/03/lib/src/ResultsWriter.cpp
 * @brief Implementation for Resultswriter.
 *

 */

#include "ResultsWriter.hpp"

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "cli.hpp"

namespace experiment03
{
namespace
{
auto safe_profile_stem(const std::string& profile_name) -> std::string
{
    namespace fs = std::filesystem;
    fs::path p(profile_name);

    std::string stem = p.stem().string();
    if (stem.empty()) stem = profile_name;
    if (stem.empty()) stem = "profile";

    for (char& c : stem)
    {
        const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
        if (!ok) c = '_';
    }

    return stem;
}

auto escape_json(const std::string& input) -> std::string
{
    std::string out;
    out.reserve(input.size());
    for (char c : input)
    {
        switch (c)
        {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
                break;
        }
    }
    return out;
}

auto now_timestamp() -> std::string
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
} // namespace

auto write_run_summary_json(const Summary& summary, std::string& out_path, std::string& out_error)
    -> bool
{
    namespace fs = std::filesystem;

    const fs::path source_results_dir =
        fs::path(__FILE__).parent_path().parent_path().parent_path() / "results";

    fs::path results_dir = source_results_dir;
    if (!fs::exists(results_dir))
    {
        results_dir = fs::path("results");
    }

    std::error_code ec;
    fs::create_directories(results_dir, ec);
    if (ec)
    {
        out_error = "failed to create results directory: " + ec.message();
        return false;
    }

    const std::string stem = now_timestamp() + "_" + safe_profile_stem(summary.profile_name);
    const fs::path out_file = results_dir / (stem + ".json");

    std::ofstream ofs(out_file);
    if (!ofs)
    {
        out_error = "failed to open output file: " + out_file.string();
        return false;
    }

    ofs << "{\n";
    ofs << "  \"profile\": \"" << escape_json(summary.profile_name) << "\",\n";
    ofs << "  \"dataset_type\": \"" << escape_json(summary.dataset_type) << "\",\n";
    ofs << "  \"autoencoder_type\": \"" << escape_json(summary.autoencoder_type) << "\",\n";
    ofs << "  \"optimizer\": {\n";
    ofs << "    \"type\": \"" << escape_json(summary.optimizer_type) << "\",\n";
    ofs << "    \"learning_rate\": " << summary.optimizer_learning_rate << ",\n";
    ofs << "    \"momentum\": " << summary.optimizer_momentum << ",\n";
    ofs << "    \"adam_beta1\": " << summary.optimizer_adam_beta1 << ",\n";
    ofs << "    \"adam_beta2\": " << summary.optimizer_adam_beta2 << ",\n";
    ofs << "    \"adam_epsilon\": " << summary.optimizer_adam_epsilon << "\n";
    ofs << "  },\n";
    ofs << "  \"exit_code\": " << summary.exit_code << ",\n";
    ofs << "  \"total_samples\": " << summary.total_samples << ",\n";
    ofs << "  \"processed_samples\": " << summary.processed_samples << ",\n";
    ofs << "  \"seen_batches\": " << summary.seen_batches << ",\n";
    ofs << "  \"epoch_mean_losses\": [";
    for (std::size_t i = 0; i < summary.epoch_mean_losses.size(); ++i)
    {
        if (i > 0) ofs << ", ";
        ofs << summary.epoch_mean_losses[i];
    }
    ofs << "],\n";
    ofs << "  \"kfold\": {\n";
    ofs << "    \"enabled\": " << (summary.kfold_enabled ? "true" : "false") << ",\n";
    ofs << "    \"n_splits\": " << summary.kfold_n_splits << ",\n";
    ofs << "    \"fold_val_losses\": [";
    for (std::size_t i = 0; i < summary.fold_val_losses.size(); ++i)
    {
        if (i > 0) ofs << ", ";
        ofs << summary.fold_val_losses[i];
    }
    ofs << "]\n";
    ofs << "  },\n";
    ofs << "  \"error\": \"" << escape_json(summary.error_message) << "\"\n";
    ofs << "}\n";

    if (!ofs.good())
    {
        out_error = "failed while writing output file: " + out_file.string();
        return false;
    }

    out_path = out_file.string();
    return true;
}

} // namespace experiment03
