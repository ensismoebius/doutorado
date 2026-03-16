#pragma once

#include <cstddef>

/**
 * Prints a concise progress summary to `std::cout`.
 * - `seen_batches`: number of batches processed so far
 * - `total_batches`: total number of expected batches
 * - `processed_samples`: number of samples processed so far
 * - `total_samples`: total number of expected samples
 * - `done`: whether the process is complete
 * Print progress in-place using a carriage return. When `done` is true the
 * function prints a trailing newline to finish the line.
 */
void printProgress(std::size_t seen_batches, std::size_t total_batches,
                   std::size_t processed_samples, std::size_t total_samples, bool done = false);
