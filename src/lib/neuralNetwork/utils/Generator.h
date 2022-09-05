#pragma once

#include "../Config.h"
#include "Rng.h"

class Generator
{
private:
public:
    Generator();
    virtual ~Generator() = 0;
    static void setNormalDistributionRandomValues(
        scalar *arr,
        const int n,
        Rng &rng,
        const scalar &mean = scalar(0),
        const scalar &sigma = scalar(1));
};
