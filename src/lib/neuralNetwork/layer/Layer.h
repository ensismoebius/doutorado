#pragma once

#include <vector>
#include <Eigen/Core>

#include "../Config.h"
#include "../utils/Rng.h"
#include "../optimizer/Optimizer.h"

class Layer
{
private:
protected:
    typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<scalar, Eigen::Dynamic, 1> Vector;

public:
    const unsigned int INPUT_SIZE;
    const unsigned int OUTPUT_SIZE;

    Layer(
        const unsigned int inputSize,
        const unsigned int outputSize);
    virtual ~Layer() = 0;

    virtual void init(const scalar &mu, scalar &sigma, Rng &rng);
    virtual void foward(const Matrix &prevLayerOutput);
    virtual const Matrix &output() const = 0;
    virtual void backprop(const Matrix &prevLayerOutput, const Matrix &netxLayerData);
    virtual const Matrix &backpropData() const = 0;

    virtual std::vector<scalar> getParameters() const = 0;
    virtual void setParameter(std::vector<scalar> param);
    virtual std::vector<scalar> getDerivative() const = 0;
};