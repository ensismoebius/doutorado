#ifndef IOPTIMIZER_H
#define IOPTIMIZER_H

#pragma once
namespace NeuralNetwork
{
namespace Optimizer{
    class IOptimizer
    {
    public:
        virtual ~IOptimizer();
        virtual void setLearningRate(float learningRate) = 0;
        virtual float activation(float value) = 0;
        virtual float activationDerivative(float value) = 0;
    };
}
}
#endif