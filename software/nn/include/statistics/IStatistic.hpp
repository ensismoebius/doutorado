#pragma once

#include <string>

namespace nn::statistics
{

class IStatistic
{
public:
    virtual void        reset()                   = 0;
    virtual void        update(float value)       = 0;
    virtual float       value()             const = 0;
    virtual std::string name()              const = 0;
    virtual ~IStatistic()                         = default;
};

} // namespace nn::statistics
