#pragma once

#include <cstddef>
#include <cstdint>

#include "Meeting01DatasetSplit.hpp"
#include "Meeting01GaFitness.hpp"
#include "Meeting01TransformerGaGenome.hpp"

namespace meeting01::ga
{

using TransformerGaIndividual = GaIndividualT<TransformerGenome>;

// Trains one Transformer-AE genome on split.train_samples, scores it on
// split.val_samples — the Transformer-family analogue of
// Meeting01GaFitness.hpp::evaluate_individual for the SNN.
void evaluate_individual(TransformerGaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total);

} // namespace meeting01::ga
