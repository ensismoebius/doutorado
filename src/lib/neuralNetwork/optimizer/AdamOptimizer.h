#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H

#pragma once

#include "IOptimizer.h"

namespace NeuralNetwork
{
    namespace Optimizer{

    class AdamOptimizer : public IOptimizer
    {
    public:
        AdamOptimizer();
         ~AdamOptimizer();
         float activation(float value);
         float activationDerivative(float value);
         void setLearningRate(float learningRate);
    private:
    };
    }
} // namespace NeuralNetwork::Optimizer

#endif