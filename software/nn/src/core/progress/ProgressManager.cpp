#include "nn/progress/ProgressManager.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <sstream>

namespace nn::progress
{

namespace
{

constexpr int kLabelWidth = 26;
constexpr int kBarWidth = 30;
constexpr int kInfoWidth = 12;
constexpr int kMetricsWidth = 12;
constexpr int kFlavorWidth = 10;

auto fit_cell(std::string text, std::size_t width) -> std::string
{
    if (text.size() > width)
    {
        if (width <= 3)
        {
            return text.substr(0, width);
        }
        return text.substr(0, width - 3) + "...";
    }
    text.append(width - text.size(), ' ');
    return text;
}

auto compact_metric_name(const std::string& name) -> std::string
{
    if (name == "train_loss") return "train";
    if (name == "val_loss") return "val";
    return name;
}

auto label_has(const std::string& label, std::string_view token) -> bool
{
    return label.find(token) != std::string::npos;
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
    oss << static_cast<int>(std::round(current)) << '/'
        << static_cast<int>(std::round(target));
    return oss.str();
}

auto format_progress_info(float progress, float current, float target) -> std::string
{
    return format_percent(progress) + " " + format_step(current, target);
}

auto format_metrics(const std::map<std::string, float>& metrics) -> std::string
{
    if (metrics.empty())
    {
        return "pending";
    }

    std::ostringstream oss;
    bool first = true;
    int metric_count = 0;
    for (const auto& [name, value] : metrics)
    {
        if (!first) oss << ' ';
        oss << compact_metric_name(name) << '=' << std::fixed << std::setprecision(3)
            << value;
        first = false;
        if (++metric_count == 1) break;
    }
    return oss.str();
}

auto format_flavor(const std::string& label, float progress, bool completed) -> std::string
{
    if (completed)
    {
        return "locked";
    }

    const bool is_lstm = label_has(label, "LSTM");
    const bool is_snn = label_has(label, "SNN");
    const bool is_run = label_has(label, "run");
    const bool is_epoch = label_has(label, "epoch");
    const bool is_batch = label_has(label, "batch");

    if (is_run)
    {
        return progress < 0.5f ? "EEG hop" : "sweeping";
    }
    if (is_lstm && is_epoch)
    {
        return progress < 0.5f ? "gating" : "settling";
    }
    if (is_lstm && is_batch)
    {
        return progress < 0.5f ? "tokens" : "holding";
    }
    if (is_snn && is_epoch)
    {
        return progress < 0.5f ? "charge" : "firing";
    }
    if (is_snn && is_batch)
    {
        return progress < 0.5f ? "spikes" : "settled";
    }
    if (is_epoch)
    {
        return progress < 0.5f ? "tuning" : "steady";
    }
    if (is_batch)
    {
        return progress < 0.5f ? "active" : "steady";
    }
    return "learning";
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
    entries_.push_back(std::move(entry));
    return id;
}

void ProgressManager::update_bar(uint32_t id, float value, const std::map<std::string, float>& metrics)
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
        std::remove_if(entries_.begin(), entries_.end(), 
            [id](const auto& e) { return e->id == id; }), 
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
                // Don't spam the console if nothing is happening
            }
            else
            {
                // Move cursor up to the start of our bar block
                if (entries_.size() > 0)
                {
                    // We print a newline at the end of each frame to stay on the same line?
                    // No, for multiple bars, we need to shift the cursor.
                    // We simulate a "frame" by printing all bars then moving cursor back up.
                }

                // Render all bars
                for (const auto& entry : entries_)
                {
                    const float current_value = entry->current_value.load();
                    const float target_value = std::max(entry->target_value, 1.0f);
                    const float progress =
                        std::clamp(current_value / target_value, 0.0f, 1.0f);
                    const int pos = static_cast<int>(kBarWidth * progress);
                    const bool completed = entry->completed.load();
                    std::map<std::string, float> metrics;

                    {
                        std::lock_guard<std::mutex> m_lock(entry->metrics_mutex);
                        metrics = entry->metrics;
                    }

                    std::stringstream ss;
                          std::string bar_cell;
                          bar_cell.reserve(static_cast<std::size_t>(kBarWidth));
                          bar_cell.push_back('[');
                          for (int i = 0; i < kBarWidth - 2; ++i)
                          {
                                bar_cell.push_back(i < pos ? '=' : ' ');
                          }
                          bar_cell.push_back(']');

                          ss << "\033[2K\r"
                              << fit_cell(entry->label, kLabelWidth)
                              << " | "
                              << fit_cell(bar_cell, kBarWidth)
                              << " | "
                              << fit_cell(format_progress_info(progress, current_value, target_value),
                                        kInfoWidth)
                              << " | "
                              << fit_cell(format_metrics(metrics), kMetricsWidth)
                              << " | "
                              << fit_cell(format_flavor(entry->label, progress, completed), kFlavorWidth);
                    
                    std::cout << ss.str() << "\n";
                }
                // Move cursor back up to the top of the block for the next frame
                std::cout << "\033[" << entries_.size() << "A";
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

} // namespace nn::progress
