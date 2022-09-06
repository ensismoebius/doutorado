#include "MaxAverage.h"

MaxAverage::MaxAverage()
{
}

MaxAverage::~MaxAverage()
{
}

scalar MaxAverage::findMaxValue(
    const scalar *x,
    const int rows,
    const int cols,
    const int columnStride,
    int &loc)
{
    // Max element in the first column
    loc = findMax(x, rows);
    scalar val = x[loc];

    x += columnStride;

    int locNext = findMax(x, rows);

    scalar valNext = x[locNext];
}

int MaxAverage::findMax(const scalar *x, int rows)
{
    return 1;
}
