#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <memory>
#include <iostream>

namespace nn::progress
{

struct ProgressEntry
{
    uint32_t id;
    std::string label;
    std::atomic<float> current_value{0.0f};
    float target_value{1.0f};
    std::atomic<bool> completed{false};
    
    mutable std::mutex metrics_mutex;
    std::map<std::string, float> metrics;

    ProgressEntry(uint32_t p_id, std::string lbl, float target) 
        : id(p_id), label(std::move(lbl)), target_value(target) {}
};

class ProgressManager
{
public:
    static ProgressManager& instance()
    {
        static ProgressManager inst;
        return inst;
    }

    ProgressManager(const ProgressManager&) = delete;
    ProgressManager& operator=(const ProgressManager&) = delete;

    uint32_t create_bar(const std::string& label, float target);
    void update_bar(uint32_t id, float value, const std::map<std::string, float>& metrics = {});
    void complete_bar(uint32_t id);
    void remove_bar(uint32_t id);
    void shutdown();

private:
    ProgressManager();
    ~ProgressManager();

    void render_loop();

    std::atomic<bool> running_{true};
    std::thread render_thread_;
    std::mutex manager_mutex_;
    std::vector<std::unique_ptr<ProgressEntry>> entries_;
    uint32_t next_id_{0};
};

} // namespace nn::progress
