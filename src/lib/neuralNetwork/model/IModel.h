#ifndef IMODEL_H
#define IMODEL_H

#pragma once

#include "../layer/Dense.h"
#include "../optimizer/IOptimizer.h"

namespace NeuralNetwork
{
    namespace Model{
    class IModel
    {
    public:
        IModel(NeuralNetwork::Layers::Dense layers[]);
        virtual ~IModel();
        virtual bool compile(char lossErrorFunc[], NeuralNetwork::Optimizer::IOptimizer optimizer) = 0;
    };
    }
}
#endif