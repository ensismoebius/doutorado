#pragma once

#include "progress/ProgressManager.hpp"

namespace nn::progress
{

class ProgressBar
{
public:
    ProgressBar(const std::string& label, float target)
    {
        id_ = ProgressManager::instance().create_bar(label, target);
    }

    ~ProgressBar()
    {
        // We don't automatically remove bars on destruction to avoid 
        // flickering if handles are short-lived. Call remove_bar explicitly if needed.
    }

    void update(float value, const std::map<std::string, float>& metrics = {})
    {
        ProgressManager::instance().update_bar(id_, value, metrics);
    }

    void mark_complete()
    {
        ProgressManager::instance().complete_bar(id_);
    }

    uint32_t id() const { return id_; }

private:
    uint32_t id_;
};

} // namespace nn::progress
