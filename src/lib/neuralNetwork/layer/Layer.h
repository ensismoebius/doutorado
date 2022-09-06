#pragma once

#include <vector>
#include <Eigen/Core>

#include "../Config.h"
#include "../utils/Rng.h"
#include "../optimizer/Optimizer.h"

class Layer
{
public:
    const unsigned int INPUT_SIZE;
    const unsigned int OUTPUT_SIZE;

    Layer(
        const unsigned int inputSize,
        const unsigned int outputSize);
    virtual ~Layer();

    virtual void init(const scalar &mean, scalar &sigma, Rng &rng) = 0;
    virtual void foward(const Matrix &prevLayerOutput) = 0;
    virtual void backprop(const Matrix &prevLayerOutput, const Matrix &netxLayerData) = 0;
    virtual void setParameter(std::vector<scalar> param) = 0;

    virtual const Matrix &output() const = 0;
    virtual const Matrix &backpropData() const = 0;

    virtual std::vector<scalar> getParameters() const = 0;
    virtual std::vector<scalar> getDerivative() const = 0;

    virtual void update(Optimizer &optimizer) const = 0;
};