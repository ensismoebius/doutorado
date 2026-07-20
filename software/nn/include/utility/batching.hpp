#ifndef BATCHING_HPP
#define BATCHING_HPP

#include <optional>
#include <span>
#include <vector>

#include "tensor/Tensor.hpp"

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
 * @param seed Semente opcional do embaralhamento. Quando fornecida, a ordem dos
 *        batches é totalmente determinística: a mesma semente produz sempre a
 *        mesma divisão. Quando omitida (padrão), semeia-se com
 *        `std::random_device` e a ordem muda a cada execução.
 * @return std::vector<Batch> Vetor de Batch, cada um contendo um par {x_batch, y_batch}.
 *
 * @note Passe uma semente em qualquer caminho cujo resultado deva ser reproduzível.
 *       A ordem dos batches é a ordem que o SGD enxerga, então um embaralhamento
 *       não semeado faz os pesos treinados mudarem de execução para execução mesmo
 *       com dados e hiperparâmetros idênticos. Ver
 *       `.wiki/Guides/Test-Quality-and-Determinism.md` (contrato de reprodutibilidade).
 */
auto create_batches(                                //
    std::span<const nn::Tensor> inputSamples,       //
    std::span<const nn::Tensor> targets,            //
    int batch_size,                                 //
    std::optional<unsigned int> seed = std::nullopt //
    ) -> std::vector<Batch>;

auto batch_to_string(const Batch& batch) -> std::string;

#endif // BATCHING_HPP
