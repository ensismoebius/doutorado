#pragma once

#include <cstddef>
#include <cstdint>

#include "Meeting01DatasetSplit.hpp"
#include "Meeting01GaFitness.hpp"
#include "Meeting01RecurrentGaGenome.hpp"

namespace meeting01::ga
{

// LSTM-AE and GRU-AE share RecurrentGenome (same tunable axes: hidden_size,
// num_layers, encoding) but are DISTINCT C++ types here on purpose: overload
// resolution on `evaluate_individual` is how the generic search driver
// (Meeting01GaSearch.hpp) knows which recurrent cell to train, without threading a
// runtime "family" string through every call site in the driver. Empty derivation —
// same layout, same genome_type, no slicing risk as long as both are always handled
// by value of their own concrete type, which the driver does (std::vector<Ind>
// throughout, never a base-class container).
struct LstmGaIndividual : GaIndividualT<RecurrentGenome>
{
};
struct GruGaIndividual : GaIndividualT<RecurrentGenome>
{
};

// Trains one LSTM-AE / GRU-AE genome on split.train_samples, scores it on
// split.val_samples — the recurrent-family analogue of
// Meeting01GaFitness.hpp::evaluate_individual for the SNN.
void evaluate_individual(LstmGaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total);

void evaluate_individual(GruGaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total);

} // namespace meeting01::ga
