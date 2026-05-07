#include "nn/progress/ProgressManager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

namespace nn::progress
{

namespace
{

constexpr int kLabelWidth = 30;
constexpr int kBarWidth = 40;
constexpr int kInfoWidth = 20;
constexpr int kMetricsWidth = 25;

void append_fitted_cell(std::ostream& os, std::string_view text, std::size_t width)
{
    if (width == 0) return;
    if (text.size() > width)
    {
        if (width <= 3)
        {
            os.write(text.data(), static_cast<std::streamsize>(width));
            return;
        }
        os.write(text.data(), static_cast<std::streamsize>(width - 3));
        os << "...";
        return;
    }
    os.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (std::size_t i = text.size(); i < width; ++i) os.put(' ');
}

auto format_percent(float progress) -> std::string
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << (progress * 100.0f) << '%';
    return oss.str();
}

auto format_step(float current, float target) -> std::string
{
    std::ostringstream oss;
    oss << static_cast<int>(std::round(current)) << '/' << static_cast<int>(std::round(target));
    return oss.str();
}

auto format_metrics(const std::map<std::string, float>& metrics) -> std::string
{
    if (metrics.empty()) return "pending";
    std::ostringstream oss;
    bool first = true;
    for (const auto& [name, value] : metrics)
    {
        if (!first) oss << ' ';
        std::string short_name = name;
        if (name == "train_loss") short_name = "train";
        else if (name == "val_loss") short_name = "val";
        oss << short_name << '=' << std::fixed << std::setprecision(3) << value;
        first = false;
    }
    return oss.str();
}

auto format_phases(const std::vector<std::string>& phases, int current_idx) -> std::string
{
    if (phases.empty()) return "";
    std::ostringstream oss;
    for (size_t i = 0; i < phases.size(); ++i)
    {
        if (i == static_cast<size_t>(current_idx)) oss << "[" << phases[i] << "]";
        else oss << phases[i];
        if (i < phases.size() - 1) oss << " → ";
    }
    return oss.str();
}

auto format_eta(int64_t start_ns, float current, float target) -> std::string
{
    if (current <= 0.0f || start_ns == 0) return "";
    const int64_t elapsed = std::chrono::steady_clock::now().time_since_epoch().count() - start_ns;
    if (elapsed <= 0) return "";
    const double time_per_item = static_cast<double>(elapsed) / current;
    const double remaining = static_cast<double>(target - current);
    if (remaining <= 0.0) return "done";
    const int64_t eta_s = static_cast<int64_t>(time_per_item * remaining) / 1'000'000'000LL;
    if (eta_s <= 0) return "";
    char buf[16];
    if (eta_s < 60)
        std::snprintf(buf, sizeof(buf), "%llds", static_cast<long long>(eta_s));
    else if (eta_s < 3600)
        std::snprintf(buf, sizeof(buf), "%lldm%02llds", static_cast<long long>(eta_s / 60), static_cast<long long>(eta_s % 60));
    else
        std::snprintf(buf, sizeof(buf), "%lldh%02lldm", static_cast<long long>(eta_s / 3600), static_cast<long long>((eta_s % 3600) / 60));
    return buf;
}

} // namespace


ProgressManager::ProgressManager()
{
    render_thread_ = std::thread(&ProgressManager::render_loop, this);
}

ProgressManager::~ProgressManager()
{
    shutdown();
}

void ProgressManager::shutdown()
{
    running_ = false;
    if (render_thread_.joinable())
    {
        render_thread_.join();
    }
}

uint32_t ProgressManager::create_bar(const std::string& label, float target)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    uint32_t id = next_id_++;
    auto entry = std::make_unique<ProgressEntry>(id, label, target);
    entry->start_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    entries_.push_back(std::move(entry));
    return id;
}

void ProgressManager::update_bar(
    uint32_t id, float value, const std::map<std::string, float>& metrics)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            entry->current_value.store(value);
            {
                std::lock_guard<std::mutex> m_lock(entry->metrics_mutex);
                entry->metrics = metrics;
            }
            return;
        }
    }
}

void ProgressManager::set_target(uint32_t id, float target)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            entry->target_value = std::max(target, 1.0f);
            return;
        }
    }
}

void ProgressManager::complete_bar(uint32_t id)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);

    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            entry->completed.store(true);
            return;
        }
    }
}

void ProgressManager::remove_bar(uint32_t id)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    entries_.erase(
        std::remove_if(
            entries_.begin(), entries_.end(), [id](const auto& e) { return e->id == id; }),
        entries_.end());
}

void ProgressManager::render_loop()
{
    while (running_)
    {
        {
            std::lock_guard<std::mutex> lock(manager_mutex_);
            if (entries_.empty())
            {
            }
            else
            {
                for (const auto& entry : entries_)
                {
                    const float current_value = entry->current_value.load();
                    const float target_value = std::max(entry->target_value, 1.0f);
                    const float progress = std::clamp(current_value / target_value, 0.0f, 1.0f);
                    const bool completed = entry->completed.load();
                    std::map<std::string, float> metrics;

                    {
                        std::lock_guard<std::mutex> m_lock(entry->metrics_mutex);
                        metrics = entry->metrics;
                    }

                    std::lock_guard<std::mutex> meta_lock(entry->metadata_mutex);

                    // Line 1: Metadata
                    std::stringstream ss_meta;
                    ss_meta << "\033[2K\r";
                    ss_meta << "\033[1m" << entry->label << "\033[0m: ";
                    append_fitted_cell(ss_meta, entry->description, kLabelWidth + 20);
                    ss_meta << " | Fold " << entry->fold_number << "/" << entry->total_folds;
                    ss_meta << " | Loss: " << (entry->loss_type.empty() ? "N/A" : entry->loss_type);
                    ss_meta << " | ";
                    append_fitted_cell(ss_meta, format_phases(entry->phases, entry->current_phase_index), 30);
                    std::cout << ss_meta.str() << "\n";

                    // Line 2: Progress bar and status
                    std::stringstream ss_prog;
                    ss_prog << "\033[2K\r";
                    ss_prog << "  Progress: [";
                    int pos = static_cast<int>(kBarWidth * progress);
                    for (int i = 0; i < kBarWidth; ++i)
                    {
                        ss_prog << (i < pos ? "█" : "░");
                    }
                    ss_prog << "] ";
                    ss_prog << format_percent(progress) << " (" << format_step(current_value, target_value) << ") ";
                    ss_prog << " | ";
                    append_fitted_cell(ss_prog, format_metrics(metrics), kMetricsWidth);
                    ss_prog << " | ";
                    const std::string eta = format_eta(entry->start_ns, current_value, target_value);
                    ss_prog << (eta.empty() ? "starting..." : "ETA: " + eta);
                    std::cout << ss_prog.str() << "\n";
                }
                // Move cursor back up by 2 * entries.size() lines
                std::cout << "\033[" << (entries_.size() * 2) << "A";
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

} // namespace nn::progress
