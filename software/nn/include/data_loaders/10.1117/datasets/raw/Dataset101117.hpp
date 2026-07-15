/**
 * @file include/nn/dataLoaders/10.1117/datasets/raw/Dataset101117.hpp
 * @brief Protocol101117dataset (migrated into datasets/raw layout).
 */

#ifndef NN_DATALOADERS_10_1117_DATASET101117_HPP
#define NN_DATALOADERS_10_1117_DATASET101117_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/loaders/EEGLoader.hpp"
#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"
#include "data_loaders/datasets/Dataset.hpp"

enum class Protocol101117InputMode
{
    Concatenated,
    EegOnly,
    AudioOnly
};

struct Protocol101117Sample
{
    nn::Tensor inputs;
    nn::Tensor targets;
    nn::Tensor audio;
    nn::Tensor eeg;
    Protocol101117InputMode input_mode = Protocol101117InputMode::Concatenated;
};

class Dataset101117 : public Dataset
{
   public:
    explicit Dataset101117(                                                        //
        std::vector<SubjectFiles> subjects,                                        //
        Protocol101117InputMode input_mode = Protocol101117InputMode::Concatenated //
    );

    void set_input_mode(Protocol101117InputMode input_mode);

    [[nodiscard]] auto input_mode() const -> Protocol101117InputMode;

    [[nodiscard]] auto get_sample(                                                //
        std::size_t idx,                                                          //
        std::optional<Protocol101117InputMode> input_mode_override = std::nullopt //
    ) const -> Protocol101117Sample;

    [[nodiscard]] auto size() const -> std::size_t override;
    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override;
    [[nodiscard]] auto collate(const std::vector<std::size_t>& indices) const -> Batch override;
    void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const override;

    [[nodiscard]] auto subjects() const -> const std::vector<SubjectFiles>&;

    void print(IDatasetPrinter& printer) const override;

   private:
    void ensureSubjectMatSessionsInitialized(std::size_t subject_index) const;

    std::vector<SubjectFiles> subjects_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::AudioSession>> audio_sessions_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::EEGSession>> eeg_sessions_;
    std::vector<std::size_t> prefix_audio_row_offsets_;
    Protocol101117InputMode input_mode_ = Protocol101117InputMode::Concatenated;
};

#endif // NN_DATALOADERS_10_1117_DATASET101117_HPP
