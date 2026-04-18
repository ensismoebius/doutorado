/**
 * @file src/core/utility/progress.cpp
 * @brief Console progress helper used by Experiment03.
 *
 * Renders a compact progress bar and counters used by the experiment runtime to
 * provide feedback while iterating over dataset batches. This implementation
 * reserves a fixed terminal area (6 log lines + 1 progress line) and redraws
 * it in-place so the bottom progress bar remains stable.
 */

#include "nn/utility/progress.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nn/logging/Logger.hpp"
#include "nn/tensor/Tensor.hpp"

void printProgress(std::size_t dataset_total_samples,
    std::size_t batch_size,
    std::size_t max_batches,
    std::size_t seen_batches,
    std::size_t processed_samples,
    bool done,
    std::size_t current_epoch,
    std::size_t total_epochs,
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

    const double ratio =
        (total_samples == 0)
            ? 0.0
            : (static_cast<double>(clamped_processed_samples) / static_cast<double>(total_samples));

    const int bar_width = 40;
    int filled = static_cast<int>(ratio * static_cast<double>(bar_width));
    if (filled < 0) filled = 0;
    if (filled > bar_width) filled = bar_width;

    std::string bar;
    bar.reserve(bar_width + 1);
    for (int i = 0; i < filled; ++i) bar.push_back('=');
    if (filled < bar_width) bar.push_back('>');
    for (int i = static_cast<int>(bar.size()); i < bar_width; ++i) bar.push_back(' ');

    const int percent = static_cast<int>(ratio * 100.0);

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
    const int reserved_lines = 7; // 6 history lines + 1 progress line
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

    // Compose progress status line
    std::ostringstream status;
    if (!context.empty())
    {
        status << context << " ";
    }
    if (current_epoch > 0 && total_epochs > 0)
    {
        status << "Epoch " << current_epoch << "/" << total_epochs << " ";
    }
    status << "Progress: [" << bar << "] " << std::setw(3) << percent << "% ("
           << clamped_seen_batches << "/" << total_batches << "b, " << clamped_processed_samples
           << "/" << total_samples << "s)";
    if (std::isfinite(current_loss))
    {
        status << "  loss: " << std::fixed << std::setprecision(6) << current_loss;
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
    // Clear and print progress on the final line.
    (*out) << "\x1b[2K" << status.str() << '\n';
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
