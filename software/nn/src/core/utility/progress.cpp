/**
 * @file src/core/utility/progress.cpp
 * @brief Console progress helper used by AutoencoderRunner.
 *
 * Renders a compact progress bar and counters used by the experiment runtime to
 * provide feedback while iterating over dataset batches. This implementation
 * reserves a fixed terminal area (6 log lines + 3 progress lines) and redraws
 * it in-place so stacked Fold/Epoch/Batch bars remain stable.
 */

#include "utility/progress.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "logging/Logger.hpp"
#include "tensor/Tensor.hpp"

namespace
{
struct ParamSnapshot
{
    std::size_t index{0};
    nn::Index rows{0};
    nn::Index cols{0};
    double sum{0.0};
    double norm{0.0};
    double mean_abs{0.0};
};

auto capture_param_snapshots(std::span<nn::Tensor*> params) -> std::vector<ParamSnapshot>
{
    std::vector<ParamSnapshot> snapshots;
    snapshots.reserve(params.size());

    for (std::size_t i = 0; i < params.size(); ++i)
    {
        nn::Tensor* p = params[i];
        if (!p) continue;

        double sum_abs = 0.0;
        for (nn::Index idx = 0; idx < p->size(); ++idx)
        {
            sum_abs += std::fabs(static_cast<double>(p->at(idx)));
        }
        const double mean_abs = (p->size() == 0) ? 0.0 : (sum_abs / static_cast<double>(p->size()));

        snapshots.push_back(ParamSnapshot{
            i,
            p->rows(),
            p->cols(),
            static_cast<double>(p->sum()),
            static_cast<double>(p->norm()),
            mean_abs,
        });
    }

    return snapshots;
}

void print_param_snapshots(const std::vector<ParamSnapshot>& snapshots)
{
    if (snapshots.empty()) return;

    std::streambuf* console_rb = nn::logging::Logger::instance().get_console_rdbuf();
    std::unique_ptr<std::ostream> console_stream;
    std::ostream* out = &std::cout;
    if (console_rb)
    {
        console_stream = std::make_unique<std::ostream>(console_rb);
        out = console_stream.get();
    }

    (*out) << "Final network parameters:\n";
    for (const auto& snapshot : snapshots)
    {
        (*out) << "  [" << snapshot.index << "] " << snapshot.rows << "x" << snapshot.cols
               << " sum=" << std::scientific << std::setprecision(6) << snapshot.sum
               << " norm=" << snapshot.norm << " mean_abs=" << snapshot.mean_abs << '\n';
    }
    out->flush();
}

class AsyncProgressDispatcher
{
   public:
    auto post(std::function<void()> task) -> void
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_worker_locked();
        pending_task_ = std::move(task);
        has_pending_task_ = true;
        cv_.notify_one();
    }

    auto flush() -> void
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !has_pending_task_ && !task_running_; });
    }

    ~AsyncProgressDispatcher()
    {
        shutdown();
    }

   private:
    auto ensure_worker_locked() -> void
    {
        if (worker_.joinable()) return;
        worker_ = std::thread([this] { run(); });
    }

    auto shutdown() -> void
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            cv_.notify_all();
        }
        if (worker_.joinable()) worker_.join();
    }

    auto run() -> void
    {
        while (true)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || has_pending_task_; });

                if (stopping_ && !has_pending_task_) break;

                task = std::move(pending_task_);
                has_pending_task_ = false;
                task_running_ = true;
            }

            if (task) task();

            {
                std::lock_guard<std::mutex> lock(mutex_);
                task_running_ = false;
            }
            cv_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::function<void()> pending_task_;
    bool has_pending_task_{false};
    bool task_running_{false};
    bool stopping_{false};
};

auto async_dispatcher() -> AsyncProgressDispatcher&
{
    static AsyncProgressDispatcher dispatcher;
    return dispatcher;
}
} // namespace

void printProgress(std::size_t dataset_total_samples,
    std::size_t batch_size,
    std::size_t max_batches,
    std::size_t seen_batches,
    std::size_t processed_samples,
    bool done,
    std::size_t current_fold,
    std::size_t total_folds,
    std::size_t current_epoch,
    std::size_t total_epochs,
    std::size_t epoch_seen_batches,
    std::size_t epoch_total_batches,
    double current_loss,
    std::span<nn::Tensor*> params,
    std::string_view context)
{
    const std::size_t safe_batch_size = (batch_size == 0) ? 1 : batch_size;
    const std::size_t dataset_total_batches =
        (dataset_total_samples + safe_batch_size - 1) / safe_batch_size;

    const std::size_t total_batches = std::min(dataset_total_batches, max_batches);
    const std::size_t total_samples = (total_batches == dataset_total_batches)
                                          ? dataset_total_samples
                                          : (total_batches * safe_batch_size);

    const std::size_t clamped_seen_batches = std::min(seen_batches, total_batches);
    const std::size_t clamped_processed_samples = std::min(processed_samples, total_samples);
    const std::size_t clamped_epoch_seen_batches =
        std::min(epoch_seen_batches, epoch_total_batches);
    const std::size_t epoch_total_samples = epoch_total_batches * safe_batch_size;
    const std::size_t epoch_seen_samples =
        std::min(clamped_epoch_seen_batches * safe_batch_size, epoch_total_samples);

    const double ratio =
        (total_samples == 0)
            ? 0.0
            : (static_cast<double>(clamped_processed_samples) / static_cast<double>(total_samples));

    const int bar_width = 40;
    const auto clamped_ratio = [](double value)
    {
        if (value < 0.0) return 0.0;
        if (value > 1.0) return 1.0;
        return value;
    };
    const auto make_bar = [](double value)
    {
        const double clamped = std::clamp(value, 0.0, 1.0);
        int filled = static_cast<int>(clamped * static_cast<double>(bar_width));
        if (filled < 0) filled = 0;
        if (filled > bar_width) filled = bar_width;

        std::string bar;
        bar.reserve(bar_width + 1);
        for (int i = 0; i < filled; ++i) bar.push_back('=');
        if (filled < bar_width) bar.push_back('>');
        for (int i = static_cast<int>(bar.size()); i < bar_width; ++i) bar.push_back(' ');

        const int percent = static_cast<int>(clamped * 100.0);
        return std::pair<std::string, int>{std::move(bar), percent};
    };

    const double epoch_batch_ratio =
        (epoch_total_batches == 0)
            ? 0.0
            : (static_cast<double>(std::min(epoch_seen_batches, epoch_total_batches)) /
                  static_cast<double>(epoch_total_batches));
    const double epoch_ratio =
        (current_epoch > 0 && total_epochs > 0)
            ? clamped_ratio((static_cast<double>(current_epoch - 1) + epoch_batch_ratio) /
                            static_cast<double>(total_epochs))
            : ratio;
    const double fold_ratio =
        (current_fold > 0 && total_folds > 0)
            ? clamped_ratio((static_cast<double>(current_fold - 1) + epoch_ratio) /
                            static_cast<double>(total_folds))
            : epoch_ratio;

    const auto [fold_bar, fold_percent] = make_bar(fold_ratio);
    const auto [epoch_bar, epoch_percent] = make_bar(epoch_ratio);
    const double batch_ratio = (epoch_total_batches > 0) ? epoch_batch_ratio : ratio;
    const auto [batch_bar, batch_percent] = make_bar(batch_ratio);

    // Drain newly emitted log lines to avoid reprocessing the same ring-buffer entries.
    auto new_lines = nn::logging::Logger::instance().drain_recent_lines();

    std::streambuf* console_rb = nn::logging::Logger::instance().get_console_rdbuf();
    std::unique_ptr<std::ostream> console_stream;
    std::ostream* out = &std::cout;
    if (console_rb)
    {
        console_stream = std::make_unique<std::ostream>(console_rb);
        out = console_stream.get();
    }

    static bool reserved_initialized = false;
    const int progress_lines = 3;
    const int reserved_lines = 6 + progress_lines; // 6 history lines + 3 progress lines
    const std::size_t history_lines = 6;
    static std::deque<std::string> local_history;

    for (const auto& ln : new_lines)
    {
        local_history.push_back(ln);
        if (local_history.size() > history_lines) local_history.pop_front();
    }

    // Build an ordered vector of exactly history_lines elements (pad top with empty lines).
    std::vector<std::string> history;
    const size_t pad =
        (history_lines > local_history.size()) ? (history_lines - local_history.size()) : 0;
    history.reserve(history_lines);
    history.resize(pad);
    std::copy(local_history.begin(), local_history.end(), std::back_inserter(history));

    // Compose stacked progress status lines.
    std::ostringstream fold_status;
    std::ostringstream epoch_status;
    std::ostringstream batch_status;

    if (!context.empty())
    {
        fold_status << context << " ";
    }
    if (current_fold > 0 && total_folds > 0)
    {
        fold_status << "Fold:  [" << fold_bar << "] " << std::setw(3) << fold_percent << "% ("
                    << current_fold << "/" << total_folds << ")";
    }
    else
    {
        fold_status << "Fold:  [" << fold_bar << "] " << std::setw(3) << fold_percent << "%";
    }

    if (current_epoch > 0 && total_epochs > 0)
    {
        epoch_status << "Epoch: [" << epoch_bar << "] " << std::setw(3) << epoch_percent << "% ("
                     << current_epoch << "/" << total_epochs << ")";
    }
    else
    {
        epoch_status << "Epoch: [" << epoch_bar << "] " << std::setw(3) << epoch_percent << "%";
    }

    if (epoch_total_batches > 0)
    {
        batch_status << "Batch: [" << batch_bar << "] " << std::setw(3) << batch_percent << "% ("
                     << clamped_epoch_seen_batches << "/" << epoch_total_batches << "b, "
                     << epoch_seen_samples << "/" << epoch_total_samples << "s)";
    }
    else
    {
        batch_status << "Batch: [" << batch_bar << "] " << std::setw(3) << batch_percent << "% ("
                     << clamped_seen_batches << "/" << total_batches << "b, "
                     << clamped_processed_samples << "/" << total_samples << "s)";
    }
    if (std::isfinite(current_loss))
    {
        batch_status << "  loss: " << std::fixed << std::setprecision(6) << current_loss;
    }

    // Initialize reserved area once, then redraw it in-place on every update.
    if (!reserved_initialized)
    {
        for (int i = 0; i < reserved_lines; ++i) (*out) << '\n';
        reserved_initialized = true;
    }
    else
    {
        (*out) << "\x1b[" << reserved_lines << "A";
    }

    // Clear and write each of the six history lines (oldest at top).
    for (size_t i = 0; i < history_lines; ++i)
    {
        (*out) << "\x1b[2K" << history[i] << '\n';
    }
    // Clear and print 3 stacked progress lines.
    (*out) << "\x1b[2K" << fold_status.str() << '\n';
    (*out) << "\x1b[2K" << epoch_status.str() << '\n';
    (*out) << "\x1b[2K" << batch_status.str() << '\n';
    out->flush();

    // If finished and a non-empty params span is provided, print final parameter summaries.
    if (done && !params.empty())
    {
        (*out) << "Final network parameters:\n";
        for (std::size_t i = 0; i < params.size(); ++i)
        {
            nn::Tensor* p = params[i];
            if (!p) continue;

            double sum_abs = 0.0;
            for (nn::Index idx = 0; idx < p->size(); ++idx)
            {
                sum_abs += std::fabs(static_cast<double>(p->at(idx)));
            }
            const double mean_abs =
                (p->size() == 0) ? 0.0 : (sum_abs / static_cast<double>(p->size()));

            (*out) << "  [" << i << "] " << p->rows() << "x" << p->cols()
                   << " sum=" << std::scientific << std::setprecision(6)
                   << static_cast<double>(p->sum()) << " norm=" << static_cast<double>(p->norm())
                   << " mean_abs=" << mean_abs << '\n';
        }
        out->flush();
    }
}

void postProgressAsync(std::size_t dataset_total_samples,
    std::size_t batch_size,
    std::size_t max_batches,
    std::size_t seen_batches,
    std::size_t processed_samples,
    bool done,
    std::size_t current_fold,
    std::size_t total_folds,
    std::size_t current_epoch,
    std::size_t total_epochs,
    std::size_t epoch_seen_batches,
    std::size_t epoch_total_batches,
    double current_loss,
    std::span<nn::Tensor*> params,
    std::string_view context)
{
    std::string context_copy(context);
    const auto snapshots =
        (done && !params.empty()) ? capture_param_snapshots(params) : std::vector<ParamSnapshot>{};

    async_dispatcher().post(
        [dataset_total_samples,
            batch_size,
            max_batches,
            seen_batches,
            processed_samples,
            done,
            current_fold,
            total_folds,
            current_epoch,
            total_epochs,
            epoch_seen_batches,
            epoch_total_batches,
            current_loss,
            context_copy = std::move(context_copy),
            snapshots]() mutable
        {
            printProgress(dataset_total_samples,
                batch_size,
                max_batches,
                seen_batches,
                processed_samples,
                done,
                current_fold,
                total_folds,
                current_epoch,
                total_epochs,
                epoch_seen_batches,
                epoch_total_batches,
                current_loss,
                std::span<nn::Tensor*>{},
                context_copy);

            if (done && !snapshots.empty())
            {
                print_param_snapshots(snapshots);
            }
        });
}

void flushProgressAsync()
{
    async_dispatcher().flush();
}
