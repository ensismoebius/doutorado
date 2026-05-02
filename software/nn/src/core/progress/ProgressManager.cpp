#include "nn/progress/ProgressManager.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <sstream>

namespace nn::progress
{

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
                    float progress = std::clamp(entry->current_value.load() / entry->target_value, 0.0f, 1.0f);
                    int bar_width = 30;
                    int pos = static_cast<int>(bar_width * progress);

                    std::stringstream ss;
                    ss << "\r" << std::left << std::setw(15) << entry->label << " [";
                    for (int i = 0; i < bar_width; ++i) ss << (i < pos ? '=' : ' ');
                    ss << "] " << (progress * 100.0f) << std::fixed << std::setprecision(1) << "% ";

                    {
                        std::lock_guard<std::mutex> m_lock(entry->metrics_mutex);
                        for (const auto& [name, val] : entry->metrics)
                        {
                            ss << "| " << name << ": " << std::fixed << std::setprecision(4) << val << " ";
                        }
                    }
                    
                    if (entry->completed) ss << " [DONE]";
                    
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
