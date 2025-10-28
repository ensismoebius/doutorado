#ifndef BATCHING_HPP
#define BATCHING_HPP

#include <vector>

#include "../tensor/Tensor.hpp"

struct Batch
{
    Tensor inputs;
    Tensor targets;
};

/**
 * @brief Divide os dados de entrada e saída em batches aleatórios.
 * Esta função embaralha as amostras de entrada e seus respectivos alvos
 * preservando a correspondência entre eles, e as divide em batches de tamanho
 * especificado. É usada para treinamento com mini-batches em redes neurais.
 *
 * @param inputSamples Tensor contendo as amostras de entrada (shape: N × D).
 * @param targets Tensor contendo os alvos correspondentes (shape: N × C).
 * @param batch_size Tamanho de cada batch.
 * @return std::vector<Batch> Vetor de Batch, cada um contendo um par {x_batch, y_batch}.
 */
auto create_batches(const std::vector<Tensor>& inputSamples, const std::vector<Tensor>& targets,
                    int batch_size) -> std::vector<Batch>;

#endif // BATCHING_HPP
