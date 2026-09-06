/**
 * @file include/data_loaders/10.1117/datasets/raw/RowRequest.hpp
 * @brief RowRequest (extracted from SynchronizedBatchAssembler.hpp).
 */

#ifndef NN_DATALOADERS_10_1117_ROWREQUEST_HPP
#define NN_DATALOADERS_10_1117_ROWREQUEST_HPP

#include <cstddef>

struct RowRequest
{
    std::size_t batch_row;
    std::size_t local_audio_row;
};

#endif // NN_DATALOADERS_10_1117_ROWREQUEST_HPP
