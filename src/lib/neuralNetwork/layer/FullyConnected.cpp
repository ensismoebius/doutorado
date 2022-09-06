#include "FullyConnected.h"
#include "../utils/Generator.h"
#include "../activation/Activation.h"

FullyConnected::FullyConnected(const unsigned int inputSize, const unsigned int outputSize)
    : Layer(3, 3)
{
}

FullyConnected::~FullyConnected()
{
}

void FullyConnected::init(const scalar &mean, scalar &sigma, Rng &rng)
{
    this->biases.resize(this->OUTPUT_SIZE);
    this->dBiases.resize(this->OUTPUT_SIZE);

    this->weights.resize(this->INPUT_SIZE, this->OUTPUT_SIZE);
    this->dWeights.resize(this->INPUT_SIZE, this->OUTPUT_SIZE);

    Generator::setNormalDistributionRandomValues(biases.data(), biases.size(), rng, mean, sigma);
    Generator::setNormalDistributionRandomValues(weights.data(), weights.size(), rng, mean, sigma);
}
void FullyConnected::foward(const Matrix &prevLayerOutput)
{
    // zValues = weigths.T * prevLayerOutput + biases
    const unsigned int inputSize = prevLayerOutput.size();
    this->zValues.resize(this->OUTPUT_SIZE, inputSize);
    this->zValues.noalias() = weights.transpose() * prevLayerOutput;
    this->zValues.colwise() += biases;

    // apply activation function
    this->activations.resize(this->OUTPUT_SIZE, inputSize);
    Activation::activate(this->zValues, this->activations);
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
    return this->dWeights;
}

std::vector<scalar> FullyConnected::getParameters() const
{
    return std::vector<scalar>();
}
std::vector<scalar> FullyConnected::getDerivative() const
{
    return std::vector<scalar>();
}

void FullyConnected::update(Optimizer &optimizer) const
{
    // ConstAlignedMapVec dw(this->dWeights.data(), this->dWeights.size());
    // ConstAlignedMapVec dw(this->biases.data(), this->biases.size());
    // AlignedMapVec w(this->weights.data(), this->weights.size());
    // AlignedMapVec w(this->biases.data(), this->biases.size());
    optimizer.update(this->dWeights, this->weights);
    optimizer.update(this->dBiases, this->biases);
}