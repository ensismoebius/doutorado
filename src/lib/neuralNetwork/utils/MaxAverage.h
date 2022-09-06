#pragma once

#include "../Config.h"

class MaxAverage
{
private:
public:
    MaxAverage();
    ~MaxAverage();

    scalar findMaxValue(
        const scalar *x,
        const int rows,
        const int cols,
        const int columnStride,
        int &loc);

    int findMax(const scalar *x, int rows);
};