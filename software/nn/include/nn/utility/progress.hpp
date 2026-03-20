#pragma once

#include <cstddef>

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
 * - `current_epoch`: optional 1-based epoch index to display
 * - `total_epochs`: optional total number of epochs to display
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
    std::size_t current_epoch = 0,
    std::size_t total_epochs = 0);
