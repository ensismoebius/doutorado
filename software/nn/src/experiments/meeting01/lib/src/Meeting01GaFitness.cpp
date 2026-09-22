#include "../include/Meeting01GaFitness.hpp"

#include <algorithm>

#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01Training.hpp"
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"

namespace meeting01::ga
{

void evaluate_individual(Meeting01GaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total)
{
    using nn::models::autoencoder::AutoencoderConfig;
    using nn::models::autoencoder::ProtocolSpikingAutoencoder;

    const Genome& g = ind.genome;

    AutoencoderConfig snn_config = to_ae_config(g, cfg);
    snn_config.initializer_seed = seed;
    snn_config.initializer_sampler_type = "ga|" + genome_key(g);

    ProtocolSpikingAutoencoder model(snn_config);

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const meeting01::TrainResult train_result = meeting01::train_with_early_stopping_snn(model,
        cfg,
        split.train_samples,
        split.val_samples,
        split.val_labels,
        g.encoding,
        g.architecture,
        g.alpha,
        g.voltage_threshold,
        seed,
        progress_index,
        progress_total,
        train_ms,
        infer_ms);

    ind.val_mse = train_result.metrics.mse;
    ind.param_count = meeting01::parameter_count(model.params());
    ind.inference_cost = meeting01::estimate_snn_macs(
        static_cast<std::size_t>(cfg.dataset.window_size), g.encoder_widths, cfg.model.time_steps);

    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {static_cast<double>(ind.val_mse), static_cast<double>(ind.inference_cost)};
}

} // namespace meeting01::ga
