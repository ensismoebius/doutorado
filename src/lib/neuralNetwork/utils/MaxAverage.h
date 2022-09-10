#pragma once

#include "../Config.h"

class MaxAverage
{
private:
    int findMax(const scalar *x, int rows);

public:
    MaxAverage();
    ~MaxAverage();

    scalar findMaxValue(
        const scalar *x,
        const int rows,
        const int cols,
        const int columnStride,
        int &loc);

    inline scalar averageBlock(
        const scalar *x, 
        const int nrow, 
        const int ncol, 
        const int colStride, 
        int &loc);

    inline scalar sumRow(const scalar *x, const int n);
};