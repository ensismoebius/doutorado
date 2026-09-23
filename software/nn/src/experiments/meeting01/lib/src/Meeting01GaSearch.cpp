// Meeting01GaSearch.hpp is header-only (the generational loop is a template,
// generalized 2026-09-22 to serve all four architecture-search families). This
// translation unit exists to EXPLICITLY INSTANTIATE the four concrete combinations
// actually used, so template errors in the shared driver surface here — in one place,
// compiled once — rather than being discovered piecemeal while compiling
// Meeting01Experiment.cpp for whichever family happens to be built first.

#include "../include/Meeting01GaSearch.hpp"

#include "../include/Meeting01RecurrentGaFitness.hpp"
#include "../include/Meeting01TransformerGaFitness.hpp"

namespace meeting01::ga
{

template GaSearchResultT<Meeting01GaIndividual> run_ga_search<Meeting01GaIndividual, GenomeBounds>(
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    const GaSearchConfigT<GenomeBounds>& ga_cfg,
    std::uint32_t base_seed,
    const EvalCallback<Meeting01GaIndividual>& on_eval);
template const Meeting01GaIndividual& pick_winner<Meeting01GaIndividual>(
    const GaSearchResultT<Meeting01GaIndividual>&);

template GaSearchResultT<LstmGaIndividual> run_ga_search<LstmGaIndividual, RecurrentGenomeBounds>(
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    const GaSearchConfigT<RecurrentGenomeBounds>& ga_cfg,
    std::uint32_t base_seed,
    const EvalCallback<LstmGaIndividual>& on_eval);
template const LstmGaIndividual& pick_winner<LstmGaIndividual>(
    const GaSearchResultT<LstmGaIndividual>&);

template GaSearchResultT<GruGaIndividual> run_ga_search<GruGaIndividual, RecurrentGenomeBounds>(
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    const GaSearchConfigT<RecurrentGenomeBounds>& ga_cfg,
    std::uint32_t base_seed,
    const EvalCallback<GruGaIndividual>& on_eval);
template const GruGaIndividual& pick_winner<GruGaIndividual>(
    const GaSearchResultT<GruGaIndividual>&);

template GaSearchResultT<TransformerGaIndividual> run_ga_search<TransformerGaIndividual,
    TransformerGenomeBounds>(const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    const GaSearchConfigT<TransformerGenomeBounds>& ga_cfg,
    std::uint32_t base_seed,
    const EvalCallback<TransformerGaIndividual>& on_eval);
template const TransformerGaIndividual& pick_winner<TransformerGaIndividual>(
    const GaSearchResultT<TransformerGaIndividual>&);

} // namespace meeting01::ga
