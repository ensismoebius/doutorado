#include <iostream>
#include "../lib/gaussian/gaussian.h"
void experiment00(){
        int start = -7;
    int end = 14;
    float step = 1.0f;
    int count = (int)(end / step);

    float x[count];
    float y1[count];
    float y2[count];
    float y3[count];

    for (int i = 0; i < count; i++)
    {
        x[i] = start + i * step;
    }

    gaussian::normal(y1, x, count, 0, 1);
    gaussian::normal(y2, x, count, 0, 2);
    gaussian::normal(y3, x, count, 3, 1);

    for(int i = 0; i < count; i++){
        std::cout << y1[i] << std::endl;
    }
}