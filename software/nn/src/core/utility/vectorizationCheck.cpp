/**
 * @file vectorizationCheck.cpp
 * @brief Prints compile-time Eigen SIMD/vectorization support information.
 */

#include "nn/utility/vectorizationCheck.hpp"

#include <Eigen/Dense>

#include "nn/logging/Logger.hpp"

void printVectorizationSupport()
{
    std::string info = "Eigen vectorization: ";

#ifdef EIGEN_VECTORIZE_AVX
    info += "AVX ";
#endif

#ifdef EIGEN_VECTORIZE_SSE4_2
    info += "SSE4.2 ";
#endif

#ifdef EIGEN_VECTORIZE_FMA
    info += "FMA ";
#endif

    NN_LOG_INFO(info);
}