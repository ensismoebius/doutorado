// TODO: Refactor this experiment into an auto-encoder training experiment which uses
// the AudioLoader and EEGLoader to load data, and trains an traditional auto-encoder model
// as well as a spiking auto-encoder model on the audio and EEG data respectively.
// Then compare, using paraconsistent features enginering techniques, to assesss the
// performance of both models in terms of the quality of the generated features vectors.
// The EEG and Audio input data must be normalized to stay between 0 and 1 before training.
// For audio and EEG the window size must be 1.5 seconds with 50% overlap.

#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <numeric> // For std::iota

#include "core/dataLoaders/10.1117/AudioData.h"
#include "core/dataLoaders/10.1117/EEGData.h"
#include "core/dataLoaders/10.1117/AudioLoader.h"
#include "core/dataLoaders/10.1117/EEGLoader.h"
#include "core/tensor/Tensor.hpp"
#include "core/layers/Module.hpp"
#include "core/layers/Linear.hpp"
#include "core/layers/Leaky.hpp"
#include "core/layers/Sequential.hpp"
#include "core/optimizers/Adam.hpp"
#include "core/optimizers/SGD.hpp"
#include "core/utility/Normalization.hpp" // Assuming this exists or will be created
#include "core/utility/Windowing.hpp"     // Assuming this exists or will be created
#include "core/paraconsistent/ParaconsistentFeatureExtractor.hpp" // Placeholder

using std::cout;
using std::string;
using namespace nn::core; // This one seems to be correct for nn::core::Tensor
using namespace nn::dataLoaders;
using namespace nn::layers;
using namespace nn::optimizers;
using namespace nn::tensor;
using namespace nn::utility; // For normalization and windowing

// Placeholder for Traditional Autoencoder
class TraditionalAutoencoder : public Module {
public:
    TraditionalAutoencoder(int input_dim, int hidden_dim)
        : encoder(std::make_unique<Sequential>(
              std::vector<std::shared_ptr<Module>>{
                  std::make_shared<Linear>(input_dim, hidden_dim),
                  // Add activation if needed, e.g., ReLU
              })),
          decoder(std::make_unique<Sequential>(
              std::vector<std::shared_ptr<Module>>{
                  std::make_shared<Linear>(hidden_dim, input_dim),
                  // Add activation if needed
              })) {}

    Tensor forward(const Tensor& input) override {
        Tensor encoded = encoder->forward(input);
        return decoder->forward(encoded);
    }

    Tensor encode(const Tensor& input) {
        return encoder->forward(input);
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_decoded = decoder->backward(grad_output);
        return encoder->backward(grad_decoded);
    }

    auto get_parameters() -> std::vector<Tensor*> override {
        std::vector<Tensor*> params;
        auto encoder_params = encoder->get_parameters();
        params.insert(params.end(), encoder_params.begin(), encoder_params.end());
        auto decoder_params = decoder->get_parameters();
        params.insert(params.end(), decoder_params.begin(), decoder_params.end());
        return params;
    }

private:
    std::unique_ptr<Sequential> encoder;
    std::unique_ptr<Sequential> decoder;
};

// Placeholder for Spiking Autoencoder
class SpikingAutoencoder : public Module {
public:
    SpikingAutoencoder(int input_dim, int hidden_dim)
        : encoder(std::make_unique<Sequential>(
              std::vector<std::shared_ptr<Module>>{
                  std::make_shared<Linear>(input_dim, hidden_dim),
                  std::make_shared<Leaky>(hidden_dim), // Spiking layer
              })),
          decoder(std::make_unique<Sequential>(
              std::vector<std::shared_ptr<Module>>{
                  std::make_shared<Linear>(hidden_dim, input_dim),
                  // Add activation if needed
              })) {}

    Tensor forward(const Tensor& input) override {
        Tensor encoded = encoder->forward(input);
        return decoder->forward(encoded);
    }

    Tensor encode(const Tensor& input) {
        return encoder->forward(input);
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_decoded = decoder->backward(grad_output);
        return encoder->backward(grad_decoded);
    }

    auto get_parameters() -> std::vector<Tensor*> override {
        std::vector<Tensor*> params;
        auto encoder_params = encoder->get_parameters();
        params.insert(params.end(), encoder_params.begin(), encoder_params.end());
        auto decoder_params = decoder->get_parameters();
        params.insert(params.end(), decoder_params.begin(), decoder_params.end());
        return params;
    }

private:
    std::unique_ptr<Sequential> encoder;
    std::unique_ptr<Sequential> decoder;
};

static void init()
{
    // Initialization code goes here
}

// Helper function to normalize data to [0, 1]
static Eigen::MatrixXf normalizeData(const Eigen::MatrixXf& data) {
    return Normalization::minMax(data);
}

// Helper function to apply windowing
static std::vector<Eigen::MatrixXf> applyWindowing(const Eigen::MatrixXf& data, float window_size_sec, float overlap_ratio, int sampling_rate) {
    return Windowing::slidingWindow(data, window_size_sec, overlap_ratio, sampling_rate);
}

static auto loadAndProcessAudio(const std::string& audioFilePath, float window_size_sec, float overlap_ratio, int sampling_rate) -> std::vector<Eigen::MatrixXf>
{
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);
    Eigen::MatrixXf audioMatrix = audioSamples.transpose(); // Convert to 1xN matrix for windowing
    Eigen::MatrixXf normalizedAudio = normalizeData(audioMatrix);
    return applyWindowing(normalizedAudio, window_size_sec, overlap_ratio, sampling_rate);
}

static auto loadAndProcessEEG(const std::string& eegFilePath, long eegRowIndex, float window_size_sec, float overlap_ratio, int sampling_rate) -> std::vector<Eigen::MatrixXf>
{
    constexpr int numChannels = 6;
    auto [eegSamplesMatrix, eegInfo] = loadEEGFromMat(eegFilePath, eegRowIndex);
    Eigen::MatrixXf normalizedEEG = normalizeData(eegSamplesMatrix);
    return applyWindowing(normalizedEEG, window_size_sec, overlap_ratio, sampling_rate);
}



static void processSubject(const std::string& subjectPath, const std::string& subjectName,
                           const std::string& audioFilePath, const std::string& eegFilePath)
{
    cout << "Processing subject: " << subjectName << '\n';

    // Define experiment parameters
    const float window_size_sec = 1.5f;
    const float overlap_ratio = 0.5f;
    const int audio_sampling_rate = 44100; // Assuming 44.1 kHz for audio
    const int eeg_sampling_rate = 1024;    // Assuming 1024 Hz for EEG

    // Load, normalize, and window audio data
    std::vector<Eigen::MatrixXf> audioWindows = loadAndProcessAudio(audioFilePath, window_size_sec, overlap_ratio, audio_sampling_rate);
    cout << "  - Loaded and processed " << audioWindows.size() << " audio windows.\n";

    // Load, normalize, and window EEG data
    // Note: eegIndex is needed for loadEEGFromMat, but we don't have it directly here.
    // Assuming for now that we can get a representative eegRowIndex, or process all.
    // For a real experiment, this would need careful synchronization.
    long representative_eeg_index = 0; // Placeholder, needs proper handling
    std::vector<Eigen::MatrixXf> eegWindows = loadAndProcessEEG(eegFilePath, representative_eeg_index, window_size_sec, overlap_ratio, eeg_sampling_rate);
    cout << "  - Loaded and processed " << eegWindows.size() << " EEG windows.\n";

    // --- Autoencoder Training ---
    // Traditional Autoencoder for Audio
    if (!audioWindows.empty()) {
        int audio_input_dim = audioWindows[0].cols(); // Assuming each window is 1 row x N cols
        int audio_hidden_dim = audio_input_dim / 2; // Example hidden dimension
        TraditionalAutoencoder audio_ae(audio_input_dim, audio_hidden_dim);
        Adam audio_optimizer(0.001); // Example learning rate
        audio_optimizer.attach(audio_ae.get_parameters());

        cout << "  - Training Traditional Autoencoder for Audio...\n";
        for (int epoch = 0; epoch < 10; ++epoch) { // Example: 10 epochs
            float epoch_loss = 0.0f;
            for (const auto& window_matrix : audioWindows) {
                Tensor input_tensor(window_matrix);
                audio_optimizer.zero_grad(audio_ae.get_parameters());
                Tensor output_tensor = audio_ae.forward(input_tensor);
                // Simple MSE loss for now
                Tensor loss_tensor = (output_tensor - input_tensor).square().mean();
                epoch_loss += loss_tensor.get_data()(0,0); // Assuming scalar loss
                Tensor grad_output = (output_tensor - input_tensor) * (2.0f / input_tensor.size());
                audio_ae.backward(grad_output);
                audio_optimizer.step(audio_ae.get_parameters());
            }
            cout << "    Epoch " << epoch << ", Loss: " << epoch_loss / audioWindows.size() << "\n";
        }
        cout << "  - Traditional Autoencoder training complete.\n";

        // Extract features
        std::vector<Tensor> audio_features;
        for (const auto& window_matrix : audioWindows) {
            audio_features.push_back(audio_ae.encode(Tensor(window_matrix)));
        }
        cout << "  - Extracted " << audio_features.size() << " audio features.\n";
    } else {
        cout << "  - No audio windows to train Traditional Autoencoder.\n";
    }


    // Spiking Autoencoder for EEG
    if (!eegWindows.empty()) {
        int eeg_input_dim = eegWindows[0].cols(); // Assuming each window is 1 row x N cols
        int eeg_hidden_dim = eeg_input_dim / 2; // Example hidden dimension
        SpikingAutoencoder eeg_sae(eeg_input_dim, eeg_hidden_dim);
        Adam eeg_optimizer(0.001); // Example learning rate
        eeg_optimizer.attach(eeg_sae.get_parameters());

        cout << "  - Training Spiking Autoencoder for EEG...\n";
        for (int epoch = 0; epoch < 10; ++epoch) { // Example: 10 epochs
            float epoch_loss = 0.0f;
            for (const auto& window_matrix : eegWindows) {
                Tensor input_tensor(window_matrix);
                eeg_optimizer.zero_grad(eeg_sae.get_parameters());
                Tensor output_tensor = eeg_sae.forward(input_tensor);
                // Simple MSE loss for now
                Tensor loss_tensor = (output_tensor - input_tensor).square().mean();
                epoch_loss += loss_tensor.get_data()(0,0); // Assuming scalar loss
                Tensor grad_output = (output_tensor - input_tensor) * (2.0f / input_tensor.size());
                eeg_sae.backward(grad_output);
                eeg_optimizer.step(eeg_sae.get_parameters());
            }
            cout << "    Epoch " << epoch << ", Loss: " << epoch_loss / eegWindows.size() << "\n";
        }
        cout << "  - Spiking Autoencoder training complete.\n";

        // Extract features
        std::vector<Tensor> eeg_features;
        for (const auto& window_matrix : eegWindows) {
            eeg_features.push_back(eeg_sae.encode(Tensor(window_matrix)));
        }
        cout << "  - Extracted " << eeg_features.size() << " EEG features.\n";

        // --- Paraconsistent Feature Engineering Comparison ---
        cout << "  - Performing Paraconsistent Feature Engineering Comparison...\n";
        // Placeholder for paraconsistent comparison logic
        // This would involve using the extracted audio_features and eeg_features
        // with a ParaconsistentFeatureExtractor or similar utility.
        // For example:
        // ParaconsistentFeatureExtractor extractor;
        // auto audio_paraconsistent_metrics = extractor.analyze(audio_features);
        // auto eeg_paraconsistent_metrics = extractor.analyze(eeg_features);
        // Compare metrics...
        cout << "  - Paraconsistent comparison placeholder executed.\n";

    } else {
        cout << "  - No EEG windows to train Spiking Autoencoder.\n";
    }
}

static void perform(const std::string& basePath)
{
    for (const auto& entry : std::filesystem::directory_iterator(basePath))
    {
        if (entry.is_directory())
        {
            string subjectPath = entry.path().string();
            string subjectName = entry.path().filename().string();
            string audioFilePath = subjectPath + "/" + subjectName + "_Audio.mat";
            string eegFilePath = subjectPath + "/" + subjectName + "_EEG.mat";

            if (std::filesystem::exists(audioFilePath) && std::filesystem::exists(eegFilePath))
            {
                processSubject(subjectPath, subjectName, audioFilePath, eegFilePath);
            }
        }
    }
}

auto main(int argc, char** argv) -> int
{
    init();

    std::string basePath =
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/";

    perform(basePath);

    return 0;
}