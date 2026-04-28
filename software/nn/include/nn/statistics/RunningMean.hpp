#pragma once

#include <string>

#include "nn/statistics/IStatistic.hpp"

namespace nn::statistics
{

class RunningMean : public IStatistic
{
public:
    explicit RunningMean(std::string name) : name_(std::move(name)) {}

    void reset() override
    {
        sum_   = 0.0F;
        count_ = 0;
    }

    void update(float value) override
    {
        sum_ += value;
        ++count_;
    }

    float value() const override
    {
        return count_ > 0 ? sum_ / static_cast<float>(count_) : 0.0F;
    }

    std::string name() const override { return name_; }

private:
    std::string name_;
    float       sum_   = 0.0F;
    int         count_ = 0;
};

} // namespace nn::statistics
