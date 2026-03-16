#include "../include/progress.hpp"

#include <iomanip>
#include <iostream>
#include <string>

void printProgress(std::size_t seen_batches, std::size_t total_batches,
                   std::size_t processed_samples, std::size_t total_samples, bool done)
{
    // Use sample-based ratio for finer-grained progress when available.
    const double ratio =
        (total_samples == 0)
            ? 0.0
            : (static_cast<double>(processed_samples) / static_cast<double>(total_samples));

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

    // Carriage return to overwrite the current console line, then flush.
    std::cout << '\r' << "Progress: [" << bar << "] " << std::setw(3) << percent << "% ("
              << seen_batches << "/" << total_batches << "b, " << processed_samples << "/"
              << total_samples << "s)";

    if (done)
    {
        std::cout << '\n';
    }
    std::cout.flush();
}
