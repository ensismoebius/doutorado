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
#include <array>
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

namespace
{

constexpr int kBarWidth = 40;
constexpr int kProgressLines = 3;
constexpr std::size_t kHistoryLines = 6;

/// The totals a run is actually measured against. They are NOT the dataset's
/// totals: a run capped by `max_batches` consumes only part of the dataset, and
/// measuring it against the whole thing would leave a finished run reading 20%
/// with the bar never filling. Pinned by
/// PrintProgress.CappedRunStillReachesOneHundredPercent.
struct EffectiveTotals
{
    std::size_t total_batches = 0;
    std::size_t total_samples = 0;
    std::size_t seen_batches = 0;      // clamped: an over-reporting loader must not print 340%
    std::size_t processed_samples = 0; // clamped, same reason
    std::size_t epoch_seen_batches = 0;
    std::size_t epoch_seen_samples = 0;
    std::size_t epoch_total_samples = 0;
    double ratio = 0.0;
};

auto compute_effective_totals(std::size_t dataset_total_samples,
    std::size_t batch_size,
    std::size_t max_batches,
    std::size_t seen_batches,
    std::size_t processed_samples,
    std::size_t epoch_seen_batches,
    std::size_t epoch_total_batches) -> EffectiveTotals
{
    // batch_size 0 is a caller error that is silently repaired to 1 rather than
    // refused. Pinned by PrintProgress.ZeroBatchSizeIsSilentlyTreatedAsOne;
    // see that test for why this is documented rather than defended.
    const std::size_t safe_batch_size = (batch_size == 0) ? 1 : batch_size;
    const std::size_t dataset_total_batches =
        (dataset_total_samples + safe_batch_size - 1) / safe_batch_size;

    EffectiveTotals totals;
    totals.total_batches = std::min(dataset_total_batches, max_batches);
    totals.total_samples = (totals.total_batches == dataset_total_batches)
                               ? dataset_total_samples
                               : (totals.total_batches * safe_batch_size);
    totals.seen_batches = std::min(seen_batches, totals.total_batches);
    totals.processed_samples = std::min(processed_samples, totals.total_samples);
    totals.epoch_seen_batches = std::min(epoch_seen_batches, epoch_total_batches);
    totals.epoch_total_samples = epoch_total_batches * safe_batch_size;
    totals.epoch_seen_samples =
        std::min(totals.epoch_seen_batches * safe_batch_size, totals.epoch_total_samples);
    totals.ratio = (totals.total_samples == 0) ? 0.0
                                               : (static_cast<double>(totals.processed_samples) /
                                                     static_cast<double>(totals.total_samples));
    return totals;
}

/// The three bars are nested, not independent: batches sit inside an epoch,
/// epochs inside a fold. Each level counts COMPLETED units of the level below
/// plus the current one's fraction — `current_epoch - 1`, not `current_epoch`.
/// Off by one here is silent: the bar still moves, just at the wrong speed.
/// Pinned by PrintProgress.EpochRatioCountsCompletedEpochsNotTheCurrentOne and
/// .FoldRatioNestsTheEpochRatioInsideIt.
struct StackedRatios
{
    double fold = 0.0;
    double epoch = 0.0;
    double batch = 0.0;
};

auto compute_stacked_ratios(const EffectiveTotals& totals,
    std::size_t current_fold,
    std::size_t total_folds,
    std::size_t current_epoch,
    std::size_t total_epochs,
    std::size_t epoch_total_batches) -> StackedRatios
{
    const auto clamp01 = [](double value) { return std::clamp(value, 0.0, 1.0); };

    const double epoch_batch_ratio = (epoch_total_batches == 0)
                                         ? 0.0
                                         : (static_cast<double>(totals.epoch_seen_batches) /
                                               static_cast<double>(epoch_total_batches));

    StackedRatios ratios;
    // Each level falls back to the level below when its counters are absent, so
    // a caller that tracks only batches still gets three consistent bars.
    ratios.epoch = (current_epoch > 0 && total_epochs > 0)
                       ? clamp01((static_cast<double>(current_epoch - 1) + epoch_batch_ratio) /
                                 static_cast<double>(total_epochs))
                       : totals.ratio;
    ratios.fold = (current_fold > 0 && total_folds > 0)
                      ? clamp01((static_cast<double>(current_fold - 1) + ratios.epoch) /
                                static_cast<double>(total_folds))
                      : ratios.epoch;
    ratios.batch = (epoch_total_batches > 0) ? epoch_batch_ratio : totals.ratio;
    return ratios;
}

/// A fixed-width bar plus its integer percent. The width is constant because
/// the display redraws in place over a reserved area: a bar that changes width
/// leaves debris on the previous frame. Pinned by
/// PrintProgress.EveryBarIsExactlyFortyCharactersWide.
auto make_bar(double value) -> std::pair<std::string, int>
{
    const double clamped = std::clamp(value, 0.0, 1.0);
    int filled =
        std::clamp(static_cast<int>(clamped * static_cast<double>(kBarWidth)), 0, kBarWidth);

    std::string bar;
    bar.reserve(kBarWidth + 1);
    bar.assign(static_cast<std::size_t>(filled), '=');
    if (filled < kBarWidth) bar.push_back('>');
    bar.resize(kBarWidth, ' ');

    return {std::move(bar), static_cast<int>(clamped * 100.0)};
}

/// "Label: [====>   ]  42%" plus, when both the index and the total are
/// positive, a "(2/5)" counter. Both must be positive: "(0/5)" or "(3/0)" would
/// be worse than saying nothing.
auto format_bar_line(std::string_view label, double ratio, std::size_t current, std::size_t total)
    -> std::string
{
    const auto [bar, percent] = make_bar(ratio);
    std::ostringstream line;
    line << label << " [" << bar << "] " << std::setw(3) << percent << "%";
    if (current > 0 && total > 0) line << " (" << current << "/" << total << ")";
    return line.str();
}

auto format_batch_line(const EffectiveTotals& totals,
    double ratio,
    std::size_t epoch_total_batches,
    double current_loss) -> std::string
{
    const auto [bar, percent] = make_bar(ratio);
    std::ostringstream line;
    line << "Batch: [" << bar << "] " << std::setw(3) << percent << "% (";
    if (epoch_total_batches > 0)
    {
        line << totals.epoch_seen_batches << "/" << epoch_total_batches << "b, "
             << totals.epoch_seen_samples << "/" << totals.epoch_total_samples << "s)";
    }
    else
    {
        line << totals.seen_batches << "/" << totals.total_batches << "b, "
             << totals.processed_samples << "/" << totals.total_samples << "s)";
    }
    // NaN is the documented "no loss yet" default. Printing "loss: nan" on every
    // step would train the reader to ignore a real NaN loss later, so a
    // non-finite loss is omitted instead. Pinned by
    // PrintProgress.ANonFiniteLossIsOmittedRatherThanPrintedAsNan.
    if (std::isfinite(current_loss))
    {
        line << "  loss: " << std::fixed << std::setprecision(6) << current_loss;
    }
    return line.str();
}

/// The six most recent log lines, oldest first, padded at the top so the block
/// is always exactly kHistoryLines tall — the redraw depends on that height.
auto padded_history(const std::deque<std::string>& recent) -> std::vector<std::string>
{
    std::vector<std::string> history;
    history.reserve(kHistoryLines);
    history.resize(kHistoryLines > recent.size() ? kHistoryLines - recent.size() : 0);
    std::copy(recent.begin(), recent.end(), std::back_inserter(history));
    return history;
}

/// Rewrites the whole reserved block in place: cursor up, then one cleared line
/// per row. The first call has nothing to move up over, so it opens the area by
/// printing the blank rows instead.
void redraw_reserved_area(std::ostream& out,
    bool& reserved_initialized,
    const std::vector<std::string>& history,
    const std::array<std::string, kProgressLines>& status_lines)
{
    const int reserved_lines = static_cast<int>(kHistoryLines) + kProgressLines;
    if (!reserved_initialized)
    {
        for (int i = 0; i < reserved_lines; ++i) out << '\n';
        reserved_initialized = true;
    }
    else
    {
        out << "\x1b[" << reserved_lines << "A";
    }

    for (const auto& entry : history) out << "\x1b[2K" << entry << '\n';
    for (const auto& entry : status_lines) out << "\x1b[2K" << entry << '\n';
    out.flush();
}

/// One line per parameter tensor, printed once at the end of a run. A null
/// pointer is skipped rather than dereferenced: the span comes from callers
/// that may hold gaps. Pinned by
/// PrintProgress.ANullParameterPointerIsSkippedRatherThanDereferenced.
void print_parameter_summary(std::ostream& out, std::span<nn::Tensor*> params)
{
    out << "Final network parameters:\n";
    for (std::size_t i = 0; i < params.size(); ++i)
    {
        nn::Tensor* param = params[i];
        if (!param) continue;

        double sum_abs = 0.0;
        for (nn::Index idx = 0; idx < param->size(); ++idx)
        {
            sum_abs += std::fabs(static_cast<double>(param->at(idx)));
        }
        const double mean_abs =
            (param->size() == 0) ? 0.0 : (sum_abs / static_cast<double>(param->size()));

        out << "  [" << i << "] " << param->rows() << "x" << param->cols()
            << " sum=" << std::scientific << std::setprecision(6)
            << static_cast<double>(param->sum()) << " norm=" << static_cast<double>(param->norm())
            << " mean_abs=" << mean_abs << '\n';
    }
    out.flush();
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
    const EffectiveTotals totals = compute_effective_totals(dataset_total_samples,
        batch_size,
        max_batches,
        seen_batches,
        processed_samples,
        epoch_seen_batches,
        epoch_total_batches);
    const StackedRatios ratios = compute_stacked_ratios(
        totals, current_fold, total_folds, current_epoch, total_epochs, epoch_total_batches);

    const std::array<std::string, kProgressLines> status_lines{
        std::string(context.empty() ? "" : std::string(context) + " ") +
            format_bar_line("Fold: ", ratios.fold, current_fold, total_folds),
        format_bar_line("Epoch:", ratios.epoch, current_epoch, total_epochs),
        format_batch_line(totals, ratios.batch, epoch_total_batches, current_loss)};

    // Drain newly emitted log lines to avoid reprocessing the same ring-buffer
    // entries. This state is static because the reserved area is a property of
    // the terminal, not of any one call.
    static std::deque<std::string> local_history;
    static bool reserved_initialized = false;
    for (auto& line : nn::logging::Logger::instance().drain_recent_lines())
    {
        local_history.push_back(std::move(line));
        if (local_history.size() > kHistoryLines) local_history.pop_front();
    }

    std::streambuf* console_rb = nn::logging::Logger::instance().get_console_rdbuf();
    std::unique_ptr<std::ostream> console_stream;
    std::ostream* out = &std::cout;
    if (console_rb)
    {
        console_stream = std::make_unique<std::ostream>(console_rb);
        out = console_stream.get();
    }

    redraw_reserved_area(*out, reserved_initialized, padded_history(local_history), status_lines);

    if (done && !params.empty()) print_parameter_summary(*out, params);
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
