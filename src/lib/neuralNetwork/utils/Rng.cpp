#include "Rng.h"
Rng::Rng(unsigned long seed) : rand(seed ? (seed & max) : 1),
                               multiplier(16807),
                               max(2147483647L)
{
}

void Rng::seed(const unsigned long seed)
{
    rand = (seed ? (seed & max) : 1);
}

double Rng::random()
{
    rand = nextRandonLong(rand);
    return double(rand) / double(max);
}

inline long Rng::nextRandonLong(long seed)
{
    unsigned long lo, hi;

    lo = multiplier * (long)(seed & 0xffff);
    hi = multiplier * (long)((unsigned long)seed >> 16);
    lo += (hi & 0x7fff) << 16;

    if (lo > max)
    {
        lo &= max;
        ++lo;
    }

    lo += hi >> 15;

    if (lo > max)
    {
        lo &= max;
        ++lo;
    }

    return (long)lo;
}