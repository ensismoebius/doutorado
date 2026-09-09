#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "Meeting01RunMetrics.hpp"

namespace meeting01
{

struct ResultRow
{
    std::string backend;
    std::string profile;
    std::string dataset;
    std::string model;
    std::string encoding;
    std::string architecture;
    int layers = 1;
    float v_th = 1.0f;
    float alpha = 0.9f;
    int run_id = 0;

    std::uint32_t seed = 0u;
    std::size_t config_hash = 0u;

    RunMetrics metrics;

    // Nested-LOSO provenance. `split` is the partition this row's metrics were
    // measured on: "val" (inner validation speaker — used for model selection and
    // early stopping) or "test" (held-out outer speaker — the headline number,
    // touched exactly once). `cv_fold` is the outer fold index, -1 for the legacy
    // pooled split. Placed after `metrics` so existing positional aggregate
    // initialisation (…, config_hash, metrics) keeps compiling.
    std::string split = "val";
    int cv_fold = -1;
};

} // namespace meeting01
