#include "gaussian.h"
#include <cmath>

namespace gaussian
{
    void normal(float *result, float sequentialArray[], int sequentialArraySize, float mu, float sigma)
    {
        for (int i = 0; i < sequentialArraySize; i++)
        {
            float p = 1.0f / (float)std::sqrt(2 * M_PI * sigma * sigma);
            result[i] = p * (float)std::pow(M_E, -0.5 / (sigma * sigma) * (sequentialArray[i] - mu) * (sequentialArray[i] - mu));
        }
    }
} // namespace gaussian
