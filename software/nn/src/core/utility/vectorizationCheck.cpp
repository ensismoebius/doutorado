/**
 * @file vectorizationCheck.cpp
 * @brief Prints compile-time xtensor SIMD/vectorization support information.
 */

#include "utility/vectorizationCheck.hpp"

#include "logging/Logger.hpp"

void printVectorizationSupport()
{
    std::string info = "xtensor SIMD: ";
#ifdef XSIMD_VERSION_MAJOR
    info += "xsimd " + std::to_string(XSIMD_VERSION_MAJOR) + "."
          + std::to_string(XSIMD_VERSION_MINOR) + " ";
#endif
#ifdef _OPENMP
    info += "OpenMP " + std::to_string(_OPENMP) + " ";
#endif
    NN_LOG_INFO(info);
}
