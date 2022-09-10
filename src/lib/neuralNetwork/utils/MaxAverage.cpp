#include "../Config.h"
#include "MaxAverage.h"

MaxAverage::MaxAverage()
{
}

MaxAverage::~MaxAverage()
{
}

inline int findMax(const scalar *x, const int n)
{
    int loc = 0;
    for (int i = 1; i < n; i++)
    {
        loc = (x[i] > x[loc]) ? i : loc;
    }
    return loc;
}

inline scalar findMaxValue(const scalar *x, const int nrow, const int ncol, const int colStride, int &loc)
{
    // Max element in the first column
    loc = findMax(x, nrow);
    scalar val = x[loc];

    x += colStride;
    int loc_next = findMax(x, nrow);
    scalar val_next = x[loc_next];
    if (val_next > val)
    {
        loc = colStride + loc_next;
        val = val_next;
    }
    if (ncol == 2)
        return val;

    // Other columns
    for (int i = 2; i < ncol; i++)
    {
        x += colStride;
        loc_next = findMax(x, nrow);
        val_next = x[loc_next];
        if (val_next > val)
        {
            loc = i * colStride + loc_next;
            val = val_next;
        }
    }

    return val;
}

inline scalar sumRow(const scalar *x, const int n)
{
    scalar c = 0;
    for (int i = 0; i < n; i++)
    {
        c += x[i];
    }
    return c;
}

inline scalar averageBlock(const scalar *x, const int nrow, const int ncol, const int colStride, int &loc)
{
    scalar sum = 0;
    for (int i = 0; i < ncol; i++)
    {
        x += colStride;
        sum += sumRow(x, nrow);
    }
    return sum / (ncol * nrow);
}