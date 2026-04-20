/**
 * @file include/nn/utility/progress.hpp
 * @brief Progress.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <string_view>

#include "nn/tensor/Tensor.hpp"

/**
 * Prints an in-place progress bar to `std::cout`.
 *
 * Effective totals are computed internally so capped runs (`max_batches`) can
 * still reach 100%.
 *
 * - `dataset_total_samples`: total number of samples in the dataset
 * - `batch_size`: configured batch size used by the loader
 * - `max_batches`: run cap (upper limit on consumed batches)
 * - `seen_batches`: number of batches processed so far
 * - `processed_samples`: number of samples processed so far
 * - `done`: whether the process is complete
 * - `current_fold`: optional 1-based fold index to display
 * - `total_folds`: optional total number of folds to display
 * - `current_epoch`: optional 1-based epoch index to display
 * - `total_epochs`: optional total number of epochs to display
 * - `epoch_seen_batches`: optional number of batches consumed in current epoch
 * - `epoch_total_batches`: optional batch budget for current epoch
 * - `current_loss`: optional current loss value to display inline (defaults to NaN)
 * - `params`: optional span of `nn::Tensor*` to print final parameter summaries when `done` is true
 * - `context`: optional short prefix label (e.g., `Fold 2/5`) shown before epoch/progress
 *
 * The display renders three stacked progress bars:
 * - Fold progress
 * - Epoch progress
 * - Batch/sample progress
 *
 * When `done` is true the function prints a trailing newline to finish the
 * progress line.
 */
void printProgress(std::size_t dataset_total_samples,
    std::size_t batch_size,
    std::size_t max_batches,
    std::size_t seen_batches,
    std::size_t processed_samples,
    bool done = false,
    std::size_t current_fold = 0,
    std::size_t total_folds = 0,
    std::size_t current_epoch = 0,
    std::size_t total_epochs = 0,
    std::size_t epoch_seen_batches = 0,
    std::size_t epoch_total_batches = 0,
    double current_loss = std::numeric_limits<double>::quiet_NaN(),
    std::span<nn::Tensor*> params = std::span<nn::Tensor*>{},
    std::string_view context = {});

/**
 * Enqueues a progress update callback for asynchronous rendering.
 *
 * The latest update is coalesced and rendered by a background worker thread,
 * minimizing I/O overhead in hot training loops.
 */
void postProgressAsync(std::size_t dataset_total_samples,
    std::size_t batch_size,
    std::size_t max_batches,
    std::size_t seen_batches,
    std::size_t processed_samples,
    bool done = false,
    std::size_t current_fold = 0,
    std::size_t total_folds = 0,
    std::size_t current_epoch = 0,
    std::size_t total_epochs = 0,
    std::size_t epoch_seen_batches = 0,
    std::size_t epoch_total_batches = 0,
    double current_loss = std::numeric_limits<double>::quiet_NaN(),
    std::span<nn::Tensor*> params = std::span<nn::Tensor*>{},
    std::string_view context = {});

/** Blocks until queued asynchronous progress updates are rendered. */
void flushProgressAsync();
