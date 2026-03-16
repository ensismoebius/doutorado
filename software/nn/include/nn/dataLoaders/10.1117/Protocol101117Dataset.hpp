#ifndef NN_DATALOADERS_10_1117_PROTOCOL101117DATASET_HPP
#define NN_DATALOADERS_10_1117_PROTOCOL101117DATASET_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/SubjectDiscovery.hpp"
#include "nn/dataLoaders/Dataset.hpp"

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

class Protocol101117Dataset : public Dataset
{
   public:
    /** Create dataset with explicit input mode. */
    explicit Protocol101117Dataset(
        std::vector<SubjectFiles> subjects,
        Protocol101117InputMode input_mode = Protocol101117InputMode::Concatenated);

    /** Set/get explicit input mode used by `get_item()` and `collate()`. */
    void set_input_mode(Protocol101117InputMode input_mode);
    [[nodiscard]] auto input_mode() const -> Protocol101117InputMode;

    /**
     * PyTorch-like sample access with per-call override for output mode.
     * @param idx Global sample index.
     * @param input_mode_override Optional override for this call only.
     * @return Protocol101117Sample containing targets and either concatenated
     *         input or separated EEG/audio tensors.
     */
    [[nodiscard]] auto get_sample(std::size_t idx,
                                  std::optional<Protocol101117InputMode> input_mode_override =
                                      std::nullopt) const -> Protocol101117Sample;

    [[nodiscard]] auto size() const -> std::size_t override;
    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override;
    [[nodiscard]] auto collate(const std::vector<std::size_t>& indices) const -> Batch override;

    [[nodiscard]] auto subjects() const -> const std::vector<SubjectFiles>&;

   private:
    /**
     * Ensure the MAT session objects for the given subject are opened and
     * initialized (lazy initialization).
     *
     * This function will create `AudioMatSession` and/or `EEGMatSession`
     * instances for the subject at `subject_index` if they are not already
     * present in the mutable session caches. It is intended to be called by
     * reader paths (e.g. `get_sample` and `collate`) before accessing
     * `audio_sessions_`/`eeg_sessions_`.
     *
     * Notes:
     * - This performs I/O (opens .mat files) and may throw on file errors.
     * - It is NOT thread-safe; callers must synchronize externally if used
     *   from multiple threads.
     */
    void ensureSubjectMatSessionsInitialized(std::size_t subject_index) const;

    std::vector<SubjectFiles> subjects_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>> audio_sessions_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>> eeg_sessions_;
    std::vector<std::size_t> prefix_audio_row_offsets_;
    Protocol101117InputMode input_mode_ = Protocol101117InputMode::Concatenated;
};

#endif // NN_DATALOADERS_10_1117_PROTOCOL101117DATASET_HPP