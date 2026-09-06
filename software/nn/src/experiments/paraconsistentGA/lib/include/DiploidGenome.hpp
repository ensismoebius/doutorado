/**
 * @file src/experiments/paraconsistentGA/lib/include/DiploidGenome.hpp
 * @brief DiploidGenome struct (extracted from GaGenome.hpp).
 */

#pragma once

#include "GaGenome.hpp"

namespace pga
{

// A diploid genotype: two haplotypes, each with its own dominance value. `expressed()`
// is the phenotype actually trained/scored — the haplotype whose dominance is not
// smaller (ties resolve to A, deterministically). The other haplotype is the recessive
// reservoir.
struct DiploidGenome
{
    Genome hap_a;
    Genome hap_b;
    float dom_a = 0.5f;
    float dom_b = 0.5f;

    [[nodiscard]] const Genome& expressed() const
    {
        return dom_a >= dom_b ? hap_a : hap_b;
    }

    bool operator==(const DiploidGenome& o) const noexcept
    {
        return hap_a == o.hap_a && hap_b == o.hap_b && dom_a == o.dom_a && dom_b == o.dom_b;
    }
};

} // namespace pga
