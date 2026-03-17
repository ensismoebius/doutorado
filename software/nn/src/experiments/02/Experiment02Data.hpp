#ifndef NN_EXPERIMENTS_02_EXPERIMENT02DATA_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02DATA_HPP

#include <string>
#include <vector>

struct EEGSample
{
    std::vector<std::vector<double>> channels;
    int modality;
    int stimulus;
    int artifacts;
};

struct AudioSample
{
    std::vector<double> signal;
    int stimulus;
    int eeg_index;
};

struct WindowedSample
{
    std::vector<double> eeg_window;
    std::vector<double> audio_window;
    int label;
};

auto load_eeg_data(const std::string& mat_path) -> std::vector<EEGSample>;
auto load_audio_data(const std::string& mat_path) -> std::vector<AudioSample>;

auto extract_windows(const std::vector<EEGSample>& eeg_samples,
    const std::vector<AudioSample>& audio_samples,
    double window_duration_sec,
    double overlap_sec,
    int eeg_rate,
    int audio_rate) -> std::vector<WindowedSample>;

void generate_synthetic_samples(std::vector<EEGSample>& eeg_samples,
    std::vector<AudioSample>& audio_samples,
    int sample_count,
    int random_seed);

#endif // NN_EXPERIMENTS_02_EXPERIMENT02DATA_HPP
