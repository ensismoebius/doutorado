#include "progress/ProgressManager.hpp"

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

constexpr int kCol1Width = 45;
constexpr int kCol2Width = 22;
constexpr int kCol3Width = 26;
constexpr int kCol4Width = 14;
constexpr int kBarWidth = kCol1Width - 2;      // "  " + bar = kCol1Width
constexpr const char* kSep = " \xe2\x94\x82 "; // " │ " (U+2502)

void append_fitted_cell(std::ostream& os, std::string_view text, std::size_t width)
{
    if (width == 0) return;
    if (text.size() > width)
    {
        if (width <= 3) //
        {
            os.write(text.data(), static_cast<std::streamsize>(width)); //
            return;                                                     //
        }
        os.write(text.data(), static_cast<std::streamsize>(width - 3)); //
        os << "...";                                                    //
        return;                                                         //
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
        if (name == "train_loss")
            short_name = "train";
        else if (name == "val_loss")
            short_name = "val";
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
        if (i == static_cast<size_t>(current_idx))
            oss << "[" << phases[i] << "]";
        else
            oss << phases[i];
        if (i < phases.size() - 1) oss << " → ";
    }
    return oss.str();
}

auto format_eta(int64_t start_ns, float current, float target, double ema_ns_per_item = 0.0)
    -> std::string
{
    if (current <= 0.0f || start_ns == 0) return "";
    const double remaining = static_cast<double>(target - current);
    if (remaining <= 0.0) return "done";
    double time_per_item = 0.0;
    if (ema_ns_per_item > 0.0)
    {
        time_per_item = ema_ns_per_item;
    }
    else
    {
        const int64_t elapsed =
            std::chrono::steady_clock::now().time_since_epoch().count() - start_ns;
        if (elapsed <= 0) return "";
        time_per_item = static_cast<double>(elapsed) / current;
    }
    const int64_t eta_s = static_cast<int64_t>(time_per_item * remaining) / 1'000'000'000LL;
    if (eta_s <= 0) return "";
    char buf[16];
    if (eta_s < 60)
        std::snprintf(buf, sizeof(buf), "%llds", static_cast<long long>(eta_s));
    else if (eta_s < 3600) //
        std::snprintf(buf, //
            sizeof(buf),
            "%lldm%02llds",
            static_cast<long long>(eta_s / 60),  //
            static_cast<long long>(eta_s % 60)); //
    else
        std::snprintf(buf, //
            sizeof(buf),
            "%lldh%02lldm",
            static_cast<long long>(eta_s / 3600),         //
            static_cast<long long>((eta_s % 3600) / 60)); //
    return buf;
}

void append_bold_cell(std::ostream& os, std::string_view text, std::size_t width)
{
    const std::size_t vis = std::min(text.size(), width);
    os << "\033[1;36m"; // bold cyan — label column, always the leftmost eye-catcher
    os.write(text.data(), static_cast<std::streamsize>(vis));
    os << "\033[0m";
    for (std::size_t i = vis; i < width; ++i) os.put(' ');
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
    if (screen_cleared_.load())
    {
        std::cout << "\033[2J\033[H";
        std::cout.flush();
    }
    // Flush accumulated log messages to stdout after bars are gone.
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (const auto& msg : messages_) std::cout << msg << "\n";
    if (!messages_.empty()) std::cout.flush();
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
    const int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            // Update EMA of ns-per-item so ETA adapts to current workload speed.
            if (entry->last_update_ns > 0)
            {
                const float delta_items = value - entry->last_update_value;
                if (delta_items > 0.0f)
                {
                    const double ns_per_item =
                        static_cast<double>(now - entry->last_update_ns) / delta_items;
                    constexpr double kAlpha = 0.3;
                    entry->ema_ns_per_item =
                        (entry->ema_ns_per_item <= 0.0)
                            ? ns_per_item
                            : kAlpha * ns_per_item + (1.0 - kAlpha) * entry->ema_ns_per_item;
                }
            }
            entry->last_update_ns = now;
            entry->last_update_value = value;

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

void ProgressManager::reset_for_epoch(uint32_t id, float target)
{
    // Sets target AND resets current_value under a single lock. Callers that
    // instead call set_target() then update_bar(0) as two separate locked ops
    // leave a window where the render thread can read the new (smaller) target
    // together with the still-stale (larger, previous-epoch) current_value —
    // producing nonsense like "16/6" for one frame.
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            entry->target_value = std::max(target, 1.0f);
            entry->current_value.store(0.0f);
            entry->last_update_ns = 0;
            entry->last_update_value = 0.0f;
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
            // Snap to 100% so the final render shows a full bar.
            entry->current_value.store(entry->target_value);
            entry->completed.store(true);
            return;
        }
    }
}

void ProgressManager::log(const std::string& msg)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    messages_.push_back(msg);
}

void ProgressManager::remove_bar(uint32_t id)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    entries_.erase(
        std::remove_if(
            entries_.begin(), entries_.end(), [id](const auto& e) { return e->id == id; }),
        entries_.end());
}

void ProgressManager::set_description(uint32_t id, const std::string& description)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            std::lock_guard<std::mutex> ml(entry->metadata_mutex);
            entry->description = description;
            return;
        }
    }
}

void ProgressManager::set_fold_info(uint32_t id, int fold_number, int total_folds)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            std::lock_guard<std::mutex> ml(entry->metadata_mutex);
            entry->fold_number = fold_number;
            entry->total_folds = total_folds;
            return;
        }
    }
}

void ProgressManager::set_loss_type(uint32_t id, const std::string& loss_type)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            std::lock_guard<std::mutex> ml(entry->metadata_mutex);
            entry->loss_type = loss_type;
            return;
        }
    }
}

void ProgressManager::set_test_loss(uint32_t id, float test_loss)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            std::lock_guard<std::mutex> ml(entry->metadata_mutex);
            entry->test_loss = test_loss;
            return;
        }
    }
}

void ProgressManager::set_phases(
    uint32_t id, const std::vector<std::string>& phases, int current_index)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            std::lock_guard<std::mutex> ml(entry->metadata_mutex);
            entry->phases = phases;
            entry->current_phase_index = current_index;
            return;
        }
    }
}

void ProgressManager::begin_active_work(uint32_t id)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            entry->active_start_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            return;
        }
    }
}

void ProgressManager::end_active_work(uint32_t id)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (auto& entry : entries_)
    {
        if (entry->id == id)
        {
            if (entry->active_start_ns != 0)
            {
                const int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
                entry->accumulated_active_ns += now - entry->active_start_ns;
                entry->active_start_ns = 0;
                ++entry->active_items_completed;
            }
            return;
        }
    }
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
                if (!screen_cleared_.exchange(true))
                {
                    // First render: clear screen so cursor is at a known origin.
                    std::cout << "\033[2J";
                }
                // Jump to absolute cursor home every frame.
                // This makes rendering immune to external stdout writes that
                // shift the cursor — the next frame always starts at (1,1).
                std::cout << "\033[H";

                // Render accumulated log messages.
                for (const auto& msg : messages_)
                {
                    std::cout << "\033[2K\r" << msg << "\n";
                }

                for (const auto& entry : entries_)
                {
                    const float cv = entry->current_value.load();
                    const float tv = std::max(entry->target_value, 1.0f);
                    const float progress = std::clamp(cv / tv, 0.0f, 1.0f);
                    std::map<std::string, float> metrics;
                    {
                        std::lock_guard<std::mutex> ml(entry->metrics_mutex);
                        metrics = entry->metrics;
                    }
                    std::lock_guard<std::mutex> meta_lock(entry->metadata_mutex);

                    // Metadata line — trim trailing empty columns so last col is unpadded.
                    {
                        std::string col2 = entry->description;
                        std::string col3;
                        if (entry->fold_number > 0 || entry->total_folds > 1)
                            col3 = "run " + std::to_string(entry->fold_number) + "/" +
                                   std::to_string(entry->total_folds);
                        if (!entry->loss_type.empty())
                            col3 += (col3.empty() ? "" : "  ") + entry->loss_type;
                        const std::string col4 =
                            format_phases(entry->phases, entry->current_phase_index);

                        const int last_col = !col4.empty()   ? 4
                                             : !col3.empty() ? 3
                                             : !col2.empty() ? 2
                                                             : 1;

                        std::stringstream ss;
                        ss << "\033[2K\r";
                        if (last_col == 1)
                        {
                            append_bold_cell(ss, entry->label, kCol1Width);
                        }
                        else
                        {
                            append_bold_cell(ss, entry->label, kCol1Width);
                            ss << kSep;
                            if (last_col == 2)
                            {
                                ss << col2;
                            }
                            else
                            {
                                append_fitted_cell(ss, col2, kCol2Width);
                            }
                        }
                        if (last_col >= 3)
                        {
                            ss << kSep;
                            if (last_col == 3)
                            {
                                ss << col3;
                            }
                            else
                            {
                                append_fitted_cell(ss, col3, kCol3Width);
                            }
                        }
                        if (last_col >= 4)
                        {
                            ss << kSep << col4;
                        }
                        std::cout << ss.str() << "\n";
                    }

                    // Progress line — always all 4 columns.
                    {
                        std::stringstream ss;
                        ss << "\033[2K\r";
                        ss << "  ";
                        const int pos = static_cast<int>(kBarWidth * progress);
                        // Green fill grows into the bar; unfilled track dims to gray so
                        // the completed portion reads clearly at a glance.
                        const char* fill_color = progress >= 1.0f ? "\033[36m" : "\033[32m";
                        ss << fill_color;
                        for (int i = 0; i < pos; ++i) ss << "█";
                        ss << "\033[90m";
                        for (int i = pos; i < kBarWidth; ++i) ss << "░";
                        ss << "\033[0m";
                        ss << kSep;
                        // Pad plain text to width FIRST, then wrap the padded result in
                        // color codes — ANSI bytes must never enter append_fitted_cell's
                        // truncation math, or a mid-escape cut leaves the terminal stuck
                        // in a color state.
                        const std::string info =
                            format_percent(progress) + "  " + format_step(cv, tv);
                        {
                            std::stringstream info_ss;
                            append_fitted_cell(info_ss, info, kCol2Width);
                            ss << "\033[1m" << info_ss.str() << "\033[0m";
                        }
                        ss << kSep;
                        {
                            std::stringstream metrics_ss;
                            append_fitted_cell(metrics_ss, format_metrics(metrics), kCol3Width);
                            ss << "\033[33m" << metrics_ss.str() << "\033[0m";
                        }
                        ss << kSep;
                        const std::string eta =
                            format_eta(entry->start_ns, cv, tv, entry->ema_ns_per_item);
                        {
                            std::stringstream eta_ss;
                            append_fitted_cell(
                                eta_ss, eta.empty() ? "starting..." : "ETA: " + eta, kCol4Width);
                            ss << "\033[35m" << eta_ss.str() << "\033[0m";
                        }
                        std::cout << ss.str() << "\n";
                    }
                }

                // Erase stale content from previous renders (handles bar removal /
                // message growth — anything left below this point is stale).
                std::cout << "\033[0J";
                std::cout.flush();

                // Auto-remove bars that were marked complete — rendered at 100%
                // this frame, safe to drop.
                entries_.erase(std::remove_if(entries_.begin(),
                                   entries_.end(),
                                   [](const auto& e) { return e->completed.load(); }),
                    entries_.end());
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

} // namespace nn::progress
