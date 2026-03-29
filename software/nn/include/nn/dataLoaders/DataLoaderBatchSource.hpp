#ifndef NN_DATALOADERS_DATALOADERBATCHSOURCE_HPP
#define NN_DATALOADERS_DATALOADERBATCHSOURCE_HPP

#include <optional>

#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/IBatchSource.hpp"

class DataLoaderBatchSource : public IBatchSource
{
   public:
    explicit DataLoaderBatchSource(DataLoader& loader) : loader_(&loader), started_(false) {}

    bool next(Batch& out) override
    {
        if (!started_)
        {
            it_.emplace(loader_->begin());
            end_.emplace(loader_->end());
            started_ = true;
        }

        if (it_->operator==(*end_))
        {
            return false;
        }

        it_->fill_batch(out);
        ++(*it_);
        return true;
    }

    void reset_epoch(std::size_t /*epoch*/) override
    {
        started_ = false;
    }

   private:
    DataLoader* loader_ = nullptr;
    std::optional<DataLoader::Iterator> it_;
    std::optional<DataLoader::Iterator> end_;
    bool started_ = false;
};

#endif // NN_DATALOADERS_DATALOADERBATCHSOURCE_HPP
