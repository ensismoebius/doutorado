#ifndef NN_DATALOADERS_IBATCHSOURCE_HPP
#define NN_DATALOADERS_IBATCHSOURCE_HPP

#include "nn/utility/batching.hpp"

class IBatchSource
{
   public:
    virtual ~IBatchSource() = default;

    // Fill `out` with the next batch. Returns true on success, false when
    // no more data is available for this epoch.
    virtual bool next(Batch& out) = 0;

    // Optional: reset source for a new epoch. Default no-op.
    virtual void reset_epoch(std::size_t /*epoch*/) {}
};

#endif // NN_DATALOADERS_IBATCHSOURCE_HPP
