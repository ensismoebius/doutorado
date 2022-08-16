#include "lib/neuralNetwork/model/Sequential.h"
#include "lib/neuralNetwork/layer/Dense.h"

int main(int argc, char const *argv[])
{
    NeuralNetwork::Layers::Dense l0(1, {1});

    NeuralNetwork::Model::Sequential model({l0});

    return 0;
}
