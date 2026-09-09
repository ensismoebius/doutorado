#include "../include/Meeting01Events.hpp"

#include <chrono>
#include <cstdio>
#include <ios>
#include <utility>

namespace meeting01
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
    path_ = path;
    warned_closed_ = false;
    if (path.empty()) return;
    out_.open(path, std::ios::out | std::ios::trunc);
}

void ExperimentEvents::reopen_append()
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (path_.empty()) return;
    if (out_.is_open()) out_.close();
    out_.open(path_, std::ios::out | std::ios::app);
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
    if (!out_.is_open())
    {
        // Observability-only sink: never throw from here (that would gate
        // training, breaking this file's core promise). But do not vanish
        // silently either — one stderr line so a broken log is noticed.
        if (!warned_closed_ && !path_.empty())
        {
            std::fprintf(stderr,
                "[meeting01-events] event sink '%s' is not open — '%s' dropped\n",
                path_.c_str(),
                type.c_str());
            warned_closed_ = true;
        }
        return;
    }
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

} // namespace meeting01
