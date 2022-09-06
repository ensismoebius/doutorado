#pragma once
#include "../Config.h"

class Activation
{
public:
    Activation();
    ~Activation();

    static void activate(Matrix &zValues, Matrix &activations);
};