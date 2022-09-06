#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <vector>

#include "../Config.h"
#include "../layer/Layer.h"
#include "../optimizer/Optimizer.h"

class FullyConnected : private Layer
{
protected:
    Matrix weights;  // m_weight
    Matrix dWeights; // m_dw
    Vector biases;   // m_bias
    Vector dBiases;  // m_db

    Matrix zValues; // m_z = prevActivations * weights + biases

    Matrix activations; // m_a

    Matrix inputDerivative; // m_din

public:
    FullyConnected(const unsigned int inputSize, const unsigned int outputSize);
    virtual ~FullyConnected();

    void init(const scalar &mean, scalar &sigma, Rng &rng);
    void foward(const Matrix &prevLayerOutput);
    void backprop(const Matrix &prevLayerOutput, const Matrix &netxLayerData);
    void setParameter(std::vector<scalar> param);

    const Matrix &output() const;
    const Matrix &backpropData() const;

    std::vector<scalar> getParameters() const = 0;
    std::vector<scalar> getDerivative() const = 0;

    void update(Optimizer &optimizer) const = 0;
};