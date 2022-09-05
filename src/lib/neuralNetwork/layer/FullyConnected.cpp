#include "FullyConnected.h"

FullyConnected::FullyConnected(const unsigned int inputSize, const unsigned int outputSize)
    : Layer(3, 3)
{
}

FullyConnected::~FullyConnected()
{
}

Layer::~Layer()
{
}

void Layer::init(const scalar &mu, scalar &sigma, Rng &rng)
{
}

void Layer::foward(const Matrix &prevLayerOutput)
{
}

const Matrix &Layer::output() const
{
    return Matrix();
}

void Layer::backprop(const Matrix &prevLayerOutput, const Matrix &netxLayerData)
{
}

const Matrix &Layer::backpropData() const
{
    return Matrix();
}

std::vector<scalar> Layer::getParameters() const
{
    return {1, 2, 3, 4};
}

void Layer::setParameter(std::vector<scalar> param)
{
}

std::vector<scalar> Layer::getDerivative() const
{
    return {1, 2, 3, 4};
}