#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#pragma once

#include "IModel.h"
#include "../layer/Dense.h"

namespace NeuralNetwork
{
namespace Model{
    class Sequential : public IModel
    {
    public:
        Sequential(NeuralNetwork::Layers::Dense layers[]);
        ~Sequential();
        bool compile(char lossErrorFunc[], NeuralNetwork::Optimizer::IOptimizer optimizer);
    };
}
}
#endif