#pragma once
// Thread-count initializer for xtensor + OpenMP.
// Replaces EigenParallel.hpp.

#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace util
{
inline void initializeXtensorParallel(int numThreads = 0)
{
#ifdef _OPENMP
    if (numThreads <= 0)
    {
        unsigned hc = std::thread::hardware_concurrency();
        numThreads = (hc > 0) ? static_cast<int>(hc) : 1;
    }
    omp_set_num_threads(numThreads);
#if _OPENMP >= 200805
    omp_set_max_active_levels(1);
#else
    omp_set_nested(0);
#endif
#else
    (void) numThreads; // no OpenMP in this build: xtensor runs single-threaded
#endif
}
} // namespace util
