#include "GuayaquilCheckpoint.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace guayaquil
{

namespace
{

constexpr int kSchemaVersion = 1;

auto sanitize(const std::string& s) -> std::string
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) out.push_back((c == '/' || c == '\\' || c == ' ') ? '-' : c);
    return out;
}

} // namespace

auto checkpoint_path(const std::filesystem::path& chk_dir, const CheckpointKey& key)
    -> std::filesystem::path
{
    char buf[512];
    std::snprintf(buf,
        sizeof(buf),
        "%s_%s_%s_%s_%s_%s_vth%.2f_a%.2f_r%02d.json",
        sanitize(key.run_tag).c_str(),
        sanitize(key.backend).c_str(),
        sanitize(key.dataset).c_str(),
        sanitize(key.model).c_str(),
        sanitize(key.encoding).c_str(),
        sanitize(key.architecture).c_str(),
        static_cast<double>(key.v_th),
        static_cast<double>(key.alpha),
        key.run_id);
    return chk_dir / buf;
}

auto checkpoint_is_valid(const std::filesystem::path& path, std::size_t expected_hash) -> bool
{
    if (!std::filesystem::exists(path)) return false;
    try
    {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        const nlohmann::json j = nlohmann::json::parse(f);
        if (j.value("version", 0) != kSchemaVersion) return false;
        const auto stored = j.value("config_hash", std::size_t{0});
        return stored == expected_hash;
    }
    catch (...)
    {
        return false;
    }
}

auto checkpoint_load(const std::filesystem::path& path) -> ResultRow
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("checkpoint_load: cannot open " + path.string());

    const nlohmann::json j = nlohmann::json::parse(f);
    const auto& m = j.at("metrics");

    ResultRow row;
    row.backend = j.value("backend", "");
    row.profile = j.value("profile", "");
    row.dataset = j.value("dataset", "");
    row.model = j.value("model", "");
    row.encoding = j.value("encoding", "");
    row.architecture = j.value("architecture", "");
    row.layers = j.value("layers", 1);
    row.v_th = j.value("v_th", 0.0f);
    row.alpha = j.value("alpha", 0.0f);
    row.run_id = j.value("run_id", 1);
    row.seed = j.value("seed", 0u);
    row.config_hash = j.value("config_hash", std::size_t{0});

    row.metrics.mse = m.value("mse", 0.0f);
    row.metrics.mae = m.value("mae", 0.0f);
    row.metrics.r2 = m.value("r2", 0.0f);
    row.metrics.precision = m.value("precision", 0.0f);
    row.metrics.recall = m.value("recall", 0.0f);
    row.metrics.f1 = m.value("f1", 0.0f);
    row.metrics.spike_rate = m.value("spike_rate", 0.0f);
    row.metrics.energy = m.value("energy", 0.0f);
    row.metrics.train_ms = m.value("train_ms", 0.0f);
    row.metrics.infer_ms = m.value("infer_ms", 0.0f);
    row.metrics.parameter_count = m.value("parameter_count", std::size_t{0});
    row.metrics.macs = m.value("macs", std::size_t{0});

    return row;
}

void checkpoint_save(const std::filesystem::path& path,
    const ResultRow& row,
    const EpochHistory& history,
    std::size_t config_hash)
{
    nlohmann::json j;
    j["version"] = kSchemaVersion;
    j["config_hash"] = config_hash;
    j["backend"] = row.backend;
    j["profile"] = row.profile;
    j["dataset"] = row.dataset;
    j["model"] = row.model;
    j["encoding"] = row.encoding;
    j["architecture"] = row.architecture;
    j["layers"] = row.layers;
    j["v_th"] = row.v_th;
    j["alpha"] = row.alpha;
    j["run_id"] = row.run_id;
    j["seed"] = row.seed;

    j["metrics"] = {
        {"mse", row.metrics.mse},
        {"mae", row.metrics.mae},
        {"r2", row.metrics.r2},
        {"precision", row.metrics.precision},
        {"recall", row.metrics.recall},
        {"f1", row.metrics.f1},
        {"spike_rate", row.metrics.spike_rate},
        {"energy", row.metrics.energy},
        {"train_ms", row.metrics.train_ms},
        {"infer_ms", row.metrics.infer_ms},
        {"parameter_count", row.metrics.parameter_count},
        {"macs", row.metrics.macs},
    };

    j["history"] = {
        {"epoch_nums", history.epoch_nums},
        {"train_losses", history.train_losses},
        {"val_losses", history.val_losses},
        {"batch_losses", history.batch_losses},
        {"batch_epochs", history.batch_epochs},
    };

    // Atomic write: write to .tmp then rename
    const auto tmp = std::filesystem::path(path.string() + ".tmp");
    {
        std::ofstream out(tmp);
        if (!out.is_open())
            throw std::runtime_error("checkpoint_save: cannot write " + tmp.string());
        out << j.dump(2);
    }
    std::filesystem::rename(tmp, path);
}

} // namespace guayaquil
