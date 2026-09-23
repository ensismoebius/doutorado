#include "../include/Meeting01TransformerGaFitness.hpp"

#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01Training.hpp"
#include "models/transformer/TransformerAutoencoder.hpp"

namespace meeting01::ga
{

void evaluate_individual(TransformerGaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total)
{
    const TransformerGenome& g = ind.genome;
    const auto tf_cfg = to_transformer_cfg(g, cfg);

    nn::models::transformer::TransformerAutoencoder model(tf_cfg);

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const meeting01::TrainResult train_result =
        meeting01::train_with_early_stopping_transformer(model,
            cfg,
            split.train_samples,
            split.val_samples,
            g.encoding,
            seed,
            progress_index,
            progress_total,
            train_ms,
            infer_ms);

    ind.val_mse = train_result.metrics.mse;
    ind.param_count = meeting01::parameter_count(model.params());
    ind.inference_cost = meeting01::estimate_transformer_macs(tf_cfg);

    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {static_cast<double>(ind.val_mse), static_cast<double>(ind.inference_cost)};
}

} // namespace meeting01::ga
