#include "Pooling.h"

template <typename Activation>
inline Pooling<Activation>::Pooling(
    const unsigned int inputWidth,
    const unsigned int inputHeight,
    const unsigned int inputChannel,
    const unsigned int poolingWidth,
    const unsigned int poolingHeight)
    : Layer(
          inputWidth * inputHeight * inputChannel,
          (inputWidth / poolingWidth) *
              (inputHeight / poolingHeight) *
              inputChannel),
      channelRows(inputHeight),
      channelCols(inputWidth),
      inputChannels(inputChannel),
      poolRows(poolingHeight),
      poolCols(poolingWidth),
      outputRows(channelRows / poolRows),
      outputCols(channelCols / poolCols)
{
}

template <typename Activation>
Pooling<Activation>::~Pooling()
{
}

template <typename Activation>
inline void Pooling<Activation>::init(const scalar &mean, scalar &sigma, Rng &rng)
{
}

template <typename Activation>
inline void Pooling<Activation>::foward(const Matrix &prevLayerOutput)
{
}

template <typename Activation>
inline void Pooling<Activation>::backprop(const Matrix &prevLayerOutput, const Matrix &netxLayerData)
{
}

template <typename Activation>
inline void Pooling<Activation>::setParameter(std::vector<scalar> param)
{
}

template <typename Activation>
inline const Matrix &Pooling<Activation>::output() const
{
}

template <typename Activation>
inline const Matrix &Pooling<Activation>::backpropData() const
{
}

template <typename Activation>
inline std::vector<scalar> Pooling<Activation>::getParameters() const
{
}

template <typename Activation>
inline std::vector<scalar> Pooling<Activation>::getDerivative() const
{
}

template <typename Activation>
inline void Pooling<Activation>::update(Optimizer &optimizer) const
{
}
