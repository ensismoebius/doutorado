#ifndef BATCHING_HPP
#define BATCHING_HPP

#include <span>
#include <vector>

#include "nn/tensor/Tensor.hpp"

/**
 * @file batching.hpp
 * @brief Simple batching utilities and the `Batch` struct.
 *
 * This is the glue between datasets/dataloaders and training loops:
 * - A `Batch` always carries a pair of tensors: `.inputs` and `.targets`.
 * - Code in this repo typically treats rows as batch dimension.
 */

struct Batch
{
    nn::Tensor inputs;
    nn::Tensor targets;
};

/**
 * @brief Divide os dados de entrada e saída em batches aleatórios.
 * Esta função embaralha as amostras de entrada e seus respectivos alvos
 * preservando a correspondência entre eles, e as divide em batches de tamanho
 * especificado. É usada para treinamento com mini-batches em redes neurais.
 *
 * @param inputSamples Span contendo as amostras de entrada (shape: N × D).
 * @param targets Span contendo os alvos correspondentes (shape: N × C).
 * @param batch_size Tamanho de cada batch.
 * @return std::vector<Batch> Vetor de Batch, cada um contendo um par {x_batch, y_batch}.
 */
auto create_batches(std::span<const nn::Tensor> inputSamples, std::span<const nn::Tensor> targets,
                    int batch_size) -> std::vector<Batch>;

#endif // BATCHING_HPP
