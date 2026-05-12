/**
 * @file src/experiments/03/lib/include/DatasetBuilder.hpp
 * @brief DatasetBuilder helper for Experiment03.
 */

#pragma once

#include <memory>
#include <vector>

#include "cli.hpp"
#include "dataLoaders/datasets/Dataset.hpp"

namespace experiment03
{
class DatasetBuilder
{
   public:
    DatasetBuilder() = default;

    DatasetBuilder& with_discovered(const std::vector<SubjectFiles>& discovered)
    {
        discovered_ = discovered;
        return *this;
    }

    DatasetBuilder& with_config(const Config& cfg)
    {
        cfg_ = &cfg;
        return *this;
    }

    auto build() -> std::shared_ptr<Dataset>;

   private:
    std::vector<SubjectFiles> discovered_;
    const Config* cfg_ = nullptr;
};

} // namespace experiment03
