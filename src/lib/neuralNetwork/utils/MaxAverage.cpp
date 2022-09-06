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
    if (valNext > val)
    {
        loc = columnStride + locNext;
        val = valNext;
    }
    if (cols == 2)
        return val;

    // Other columns
    for (int i = 2; i < cols; i++)
    {
        x += columnStride;
        locNext = findMax(x, rows);
        valNext = x[locNext];
        if (valNext > val)
        {
            loc = i * locNext + locNext;
            val = valNext;
        }
    }
    return val;
}

int MaxAverage::findMax(const scalar *x, int rows)
{
    int loc = 0;
    for (int i = 1; i < rows; i++)
    {
        loc = (x[i] > x[loc]) ? i : loc;
    }

    return loc;
}
