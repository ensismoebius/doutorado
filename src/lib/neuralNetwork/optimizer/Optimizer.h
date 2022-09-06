#pragma once
#include "../Config.h"

class Optimizer
{
private:
public:
    Optimizer();
    ~Optimizer();
    void update(const Matrix &dWeights, const Matrix &weights);
};
