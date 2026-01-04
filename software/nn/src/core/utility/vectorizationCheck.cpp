#include "vectorizationCheck.hpp"

#include <iostream>

#include "core/tensor/EigenTensorBackend.hpp"

void printVectorizationSupport()
{
    std::cout << "Eigen vectorization: ";

#ifdef EIGEN_VECTORIZE_AVX
    std::cout << "AVX ";
#endif

#ifdef EIGEN_VECTORIZE_SSE4_2
    std::cout << "SSE4.2 ";
#endif

#ifdef EIGEN_VECTORIZE_FMA
    std::cout << "FMA ";
#endif

    std::cout << '\n' << '\n';
}