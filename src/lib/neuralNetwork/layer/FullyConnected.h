#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <vector>

#include "../Config.h"
#include "../layer/Layer.h"

class FullyConnected : public Layer
{
private:

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
};