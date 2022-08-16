#include "AdamOptimizer.h"

NeuralNetwork::Optimizer::AdamOptimizer::AdamOptimizer()
{
}

NeuralNetwork::Optimizer::AdamOptimizer::~AdamOptimizer()
{
}

float NeuralNetwork::Optimizer::AdamOptimizer::activation(float value)
{
    return 1.0;
}
float NeuralNetwork::Optimizer::AdamOptimizer::activationDerivative(float value)
{
    return 1.0;
}
void NeuralNetwork::Optimizer::AdamOptimizer::setLearningRate(float learningRate)
{
}