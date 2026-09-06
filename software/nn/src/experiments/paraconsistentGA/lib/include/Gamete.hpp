/**
 * @file src/experiments/paraconsistentGA/lib/include/Gamete.hpp
 * @brief Gamete struct (extracted from GaGenome.hpp).
 */

#pragma once

#include "GaGenome.hpp"

namespace pga
{

// One haploid gamete produced by meiosis: a recombined+mutated haplotype and the
// dominance value it will carry into the child.
struct Gamete
{
    Genome haplotype;
    float dominance = 0.5f;
};

} // namespace pga
