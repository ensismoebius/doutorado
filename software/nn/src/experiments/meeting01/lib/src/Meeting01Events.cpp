#include "../include/GuayaquilEvents.hpp"

#include <chrono>
#include <ios>
#include <utility>

namespace guayaquil
{

auto ExperimentEvents::instance() -> ExperimentEvents&
{
    static ExperimentEvents inst;
    return inst;
}

ExperimentEvents::~ExperimentEvents()
{
    close();
}

void ExperimentEvents::open(const std::string& path)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (out_.is_open()) out_.close();
    if (path.empty()) return;
    out_.open(path, std::ios::out | std::ios::trunc);
}

auto ExperimentEvents::is_open() const -> bool
{
    const std::lock_guard<std::mutex> lock(mu_);
    return out_.is_open();
}

void ExperimentEvents::set_common(nlohmann::json common)
{
    const std::lock_guard<std::mutex> lock(mu_);
    common_ = std::move(common);
}

void ExperimentEvents::set_pending_context(EventContext ctx)
{
    const std::lock_guard<std::mutex> lock(mu_);
    pending_ = std::move(ctx);
}

auto ExperimentEvents::pending_context() const -> EventContext
{
    const std::lock_guard<std::mutex> lock(mu_);
    return pending_;
}

void ExperimentEvents::emit(const std::string& type, const nlohmann::json& fields)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (!out_.is_open()) return;
    nlohmann::json line = common_;
    line["ts_unix"] =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    line["type"] = type;
    if (fields.is_object())
        for (const auto& [key, value] : fields.items()) line[key] = value;
    out_ << line.dump() << '\n';
    out_.flush();
}

void ExperimentEvents::close()
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (out_.is_open()) out_.close();
}

} // namespace guayaquil
