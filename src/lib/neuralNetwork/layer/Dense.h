#ifndef MODEL_H
#define MODEL_H

#pragma once
namespace NeuralNetwork
{

    namespace Layers{
    class Dense
    {
    private:
    public:
        Dense(unsigned int units, unsigned int shape[]);
        ~Dense();
    };
    }
}

#endif