/**
 * @file src/experiments/thesis/lib/include/IndexIntersection.hpp
 * @brief Order-independent intersection of two sample-index lists.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

namespace thesis
{

/**
 * The indices present in both `a` and `b`, ascending, whatever order the inputs are in.
 *
 * Regression guard (2026-11-12): this used to be a bare std::set_intersection, whose
 * precondition is that BOTH ranges are sorted. A fold's train indices are the other folds'
 * buckets concatenated ([fold 1 subjects][fold 2 subjects]...), each ascending but the whole
 * not, so the intersection silently dropped every sample of a later bucket whose index was
 * below the end of an earlier one. Simulated on the thesis dataset's real per-speaker sizes,
 * flat-CV training sets lost 62% of their samples on average (15-78%), often keeping 3 of
 * 12 training speakers; nested-CV inner folds lost samples the same way, down to a
 * validation fold with a single speaker. Nothing failed.
 */
inline auto intersect_indices(std::vector<std::size_t> a, std::vector<std::size_t> b)
    -> std::vector<std::size_t>
{
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<std::size_t> out;
    out.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}

} // namespace thesis
