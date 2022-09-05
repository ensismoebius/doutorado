#include "FullyConnected.h"

FullyConnected::FullyConnected(const unsigned int inputSize, const unsigned int outputSize)
    : Layer(3, 3)
{
}

FullyConnected::~FullyConnected()
{
}

void FullyConnected::init(const scalar &mean, scalar &sigma, Rng &rng)
{
    this->weights.resize(this->INPUT_SIZE, this->OUTPUT_SIZE);
    this->biases.resize(this->OUTPUT_SIZE);
    this->dWeights.resize(this->INPUT_SIZE, this->OUTPUT_SIZE);
    this->dBiases.resize(this->OUTPUT_SIZE);
}
void FullyConnected::foward(const Matrix &prevLayerOutput)
{
}
void FullyConnected::backprop(const Matrix &prevLayerOutput, const Matrix &netxLayerData)
{
}
void FullyConnected::setParameter(std::vector<scalar> param)
{
}

const Matrix &FullyConnected::output() const
{
    return Matrix();
}
const Matrix &FullyConnected::backpropData() const
{
    return Matrix();
}

std::vector<scalar> FullyConnected::getParameters() const
{
    return std::vector<scalar>();
}
std::vector<scalar> FullyConnected::getDerivative() const
{
    return std::vector<scalar>();
}