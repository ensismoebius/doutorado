#pragma once

#include <vector>
#include <stdexcept>
#include <Eigen/Core>

#include "../Config.h"
#include "../layer/Layer.h"

template <typename Activation>
class Pooling : public Layer
{
private:
    const unsigned int channelRows;   // m_channel_rows
    const unsigned int channelCols;   // m_channel_cols
    const unsigned int inputChannels; // m_in_channel
    const unsigned int poolRows;      // m_pool_rows
    const unsigned int poolCols;      // m_pool_cols
    const unsigned int outputRows;    // m_out_rows
    const unsigned int outputCols;    // m_out_cols

    IntegerMatrix maxLocation; // _m_loc
    Matrix zmatrix;            // m_z;
    Matrix activationMatrix;   // m_a
    Matrix inputDerivative;    // m_din

public:
    Pooling(
        const unsigned int inputWidth,
        const unsigned int inputHeight,
        const unsigned int inputChannel,
        const unsigned int poolingWidth,
        const unsigned int poolingHeight);
    ~Pooling();

    void init(const scalar &mean, scalar &sigma, Rng &rng);

    void foward(const Matrix &prevLayerOutput);
    void backprop(const Matrix &prevLayerOutput, const Matrix &netxLayerData);
    void setParameter(std::vector<scalar> param);

    const Matrix &output() const;
    const Matrix &backpropData() const;

    std::vector<scalar> getParameters() const;
    std::vector<scalar> getDerivative() const;

    void update(Optimizer &optimizer) const;
};