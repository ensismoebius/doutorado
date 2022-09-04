#include "Layer.h"
Layer::Layer( 
    const unsigned int inputSize,
    const unsigned int outputSize)
    : INPUT_SIZE(inputSize),
      OUTPUT_SIZE(outputSize)
{

}

void Layer::init(const scalar &mu, scalar &sigma, Rng &rng) {
    
}

void Layer::foward(const Matrix &prevLayerOutput) {
    
}

void Layer::backprop(const Matrix &prevLayerOutput, const Matrix &netxLayerData) {
    
}

void Layer::setParameter(std::vector<scalar> param) {
    
}
