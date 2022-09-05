#include <cmath>
#include "Generator.h"
#include "Rng.h"

Generator::Generator()
{
}
void Generator::setNormalDistributionRandomValues(
    scalar *arr,
    const int n,
    Rng &rng,
    const scalar &mean,
    const scalar &sigma)
{
    const double twoPi = M_PI * 2;

    for (int i = 0; i < n - 1; i += 2)
    {
        const double t1 = sigma * std::sqrt(-2 * std::log(rng.random()));
        const double t2 = twoPi * rng.random();
        arr[i] = t1 * std::cos(t2) + mean;
        arr[i + 1] = t1 * std::sin(t2) + mean;
    }

    if(n % 2 == 1){
        const double t1 = sigma * std::sqrt(-2 * std::log(rng.random()));
        const double t2 = twoPi * rng.random();
        arr[n - 1] = t1 * std::cos(t2) + mean;
    }
}
