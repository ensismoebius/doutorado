#pragma once

#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    int64_t start_ns{0}; // set once in create_bar(); nanoseconds since steady_clock epoch
    int64_t active_start_ns{0}; // timestamp when actively working (excluding cached/skipped)
    int64_t accumulated_active_ns{0}; // total active nanoseconds for accurate ETA
    int32_t active_items_completed{0}; // number of items that required actual work
    int64_t last_update_ns{0};   // timestamp of most recent update_bar call
    float   last_update_value{0.0f}; // value at most recent update_bar call
    double  ema_ns_per_item{0.0};    // EMA of nanoseconds per item (alpha=0.3)

    // Metadata fields (protected by metadata_mutex)
    mutable std::mutex metadata_mutex;
    std::string description;                           // e.g., "LSTM Autoencoder", "SNN"
    int fold_number{0};                                // current fold/run number
    int total_folds{1};                                // total folds/runs
    std::string loss_type;                             // e.g., "MSE", "CrossEntropy"
    float test_loss{0.0f};                             // test loss value
    std::vector<std::string> phases;                   // e.g., ["LSTM", "SNN", "LSTM"]
    int current_phase_index{0};                        // index into phases array
    mutable std::atomic<bool> metadata_printed{false}; // track if metadata has been rendered

    ProgressEntry(uint32_t p_id, std::string lbl, float target)
        : id(p_id), label(std::move(lbl)), target_value(target)
    {
    }
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
    void set_target(uint32_t id, float target);
    void complete_bar(uint32_t id);
    void remove_bar(uint32_t id);
    void shutdown();

    void begin_active_work(uint32_t id);
    void end_active_work(uint32_t id);

    // Thread-safe message log — rendered above bars; flushed to stdout on shutdown.
    void log(const std::string& msg);

    // Metadata setters
    void set_description(uint32_t id, const std::string& description);
    void set_fold_info(uint32_t id, int fold_number, int total_folds);
    void set_loss_type(uint32_t id, const std::string& loss_type);
    void set_test_loss(uint32_t id, float test_loss);
    void set_phases(uint32_t id, const std::vector<std::string>& phases, int current_index = 0);

   private:
    ProgressManager();
    ~ProgressManager();

    void render_loop();

    std::atomic<bool> running_{true};
    std::atomic<bool> screen_cleared_{false};
    std::thread render_thread_;
    std::mutex manager_mutex_;
    std::vector<std::unique_ptr<ProgressEntry>> entries_;
    std::vector<std::string> messages_;
    uint32_t next_id_{0};
};

} // namespace nn::progress
