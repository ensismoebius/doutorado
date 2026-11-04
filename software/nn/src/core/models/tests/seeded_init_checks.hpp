#ifndef NN_CORE_MODELS_TESTS_SEEDED_INIT_CHECKS_HPP
#define NN_CORE_MODELS_TESTS_SEEDED_INIT_CHECKS_HPP

// seeded_init_checks.hpp -- the init_seed contract shared by LSTMAutoencoder, GRUAutoencoder
// and TransformerAutoencoder (their *_autoencoder_gtest.cpp call it):
//
//   init_seed unset   two models are identical (the fixed seeds in the model code)
//   init_seed = 42    two models are identical (deterministic)
//   42 vs 43          every RANDOMLY initialised tensor differs; constant ones (zero biases,
//                     the LSTM's forget-gate 1s, LayerNorm's 1 / 0) are the same
//   one seeded model  no two random tensors of the same shape are equal: stacked layers of
//                     one shape no longer start identical (they did with the fixed seeds)
//
// Until 2026-11-04 (meeting01 results_format 9) these models had no seed at all, so meeting01's
// "two seeds" trained them twice from the same weights.

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "tensor/Tensor.hpp"

namespace nn::test
{

inline auto same_values(const nn::Tensor& a, const nn::Tensor& b) -> bool
{
    if (a.size() != b.size() || a.get_shape() != b.get_shape()) return false;
    for (nn::Index k = 0; k < static_cast<nn::Index>(a.size()); ++k)
        if (a.at(k) != b.at(k)) return false;
    return true;
}

// Random = holds a value other than 0 and 1 (the constants every initialiser here uses).
inline auto randomly_initialised(const nn::Tensor& t) -> bool
{
    for (nn::Index k = 0; k < static_cast<nn::Index>(t.size()); ++k)
        if (t.at(k) != 0.0f && t.at(k) != 1.0f) return true;
    return false;
}

template <typename Model, typename Config>
void expect_init_seed_contract(Config cfg)
{
    cfg.init_seed.reset();
    const auto fixed_a = Model(cfg).state_dict();
    const auto fixed_b = Model(cfg).state_dict();
    cfg.init_seed = 42u;
    const auto s42 = Model(cfg).state_dict();
    const auto s42_again = Model(cfg).state_dict();
    cfg.init_seed = 43u;
    const auto s43 = Model(cfg).state_dict();

    ASSERT_EQ(s42.size(), fixed_a.size());
    int n_random = 0;
    for (const auto& [name, t42] : s42)
    {
        EXPECT_TRUE(same_values(fixed_a.at(name), fixed_b.at(name))) << name << ": unset not fixed";
        EXPECT_TRUE(same_values(t42, s42_again.at(name))) << name << ": seed 42 not deterministic";
        if (randomly_initialised(t42))
        {
            ++n_random;
            EXPECT_FALSE(same_values(t42, s43.at(name))) << name << ": seeds 42 and 43 agree";
            EXPECT_FALSE(same_values(t42, fixed_a.at(name)))
                << name << ": seed 42 = the fixed init";
        }
        else
        {
            EXPECT_TRUE(same_values(t42, s43.at(name))) << name << ": a constant tensor changed";
        }
    }
    EXPECT_GT(n_random, 0);

    for (auto i = s42.begin(); i != s42.end(); ++i)
        for (auto j = std::next(i); j != s42.end(); ++j)
        {
            if (randomly_initialised(i->second) && i->second.get_shape() == j->second.get_shape())
            {
                EXPECT_FALSE(same_values(i->second, j->second))
                    << i->first << " and " << j->first << " start identical under one seed";
            }
        }
}

} // namespace nn::test

#endif // NN_CORE_MODELS_TESTS_SEEDED_INIT_CHECKS_HPP
