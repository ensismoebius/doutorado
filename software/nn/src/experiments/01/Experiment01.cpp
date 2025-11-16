// TODO: Refactor this experiment into an auto-encoder training experiment which uses
// the AudioLoader and EEGLoader to load data, and trains an traditional auto-encoder model
// as well as a spiking auto-encoder model on the audio and EEG data respectively.
// Then compare, using paraconsistent features enginering techniques, to assesss the
// performance of both models in terms of the quality of the generated features vectors.
// The EEG and Audio input data must be normalized to stay between 0 and 1 before training.
// For audio and EEG the window size must be 1.5 seconds with 50% overlap.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/NetworkSerializer.hpp"
#include "core/dataLoaders/10.1117/AudioLoader.h"
#include "core/dataLoaders/10.1117/EEGLoader.h"
#include "core/layers/Leaky.hpp"
#include "core/layers/Linear.hpp"
#include "core/layers/Module.hpp"
#include "core/layers/Sequential.hpp"
#include "core/optimizers/Adam.hpp"
#include "core/tensor/Tensor.hpp"
#include "core/utility/Normalization.hpp"
#include "core/utility/Windowing.hpp"

using nn::dataLoaders::loadAudioFromMat;
using nn::dataLoaders::loadEEGFromMat;
using std::cout;
using std::initializer_list;
using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::string;

class TraditionalAutoencoder : public Module
{
   private:
    std::unique_ptr<Sequential> encoder;
    std::unique_ptr<Sequential> decoder;

   public:
    TraditionalAutoencoder(int input_dim, int hidden_dim)
        : encoder(make_unique<Sequential>( // Encoder
              initializer_list<shared_ptr<Module>>{
                  // Layers
                  make_shared<Linear>(input_dim, hidden_dim), //  hidden layer
                  make_shared<Leaky>(hidden_dim, 0.01F),      // ajustado para assinatura hipotética
              })),
          decoder(make_unique<Sequential>( // Decoder
              initializer_list<shared_ptr<Module>>{
                  // Layers
                  make_shared<Linear>(hidden_dim, input_dim), // Reconstruction layer
                  // output activation left out; assume reconstruction MSE on [0,1]
              }))
    {
    }

    auto forward(const Tensor& input) -> Tensor override
    {
        Tensor encoded = encoder->forward(input);
        return decoder->forward(encoded);
    }

    auto encode(const Tensor& input) -> Tensor
    {
        return encoder->forward(input);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        Tensor grad_decoded = decoder->backward(grad_output);
        return encoder->backward(grad_decoded);
    }

    auto params() -> std::vector<Tensor*> override
    {
        std::vector<Tensor*> params;
        auto encoder_params = encoder->params();
        auto decoder_params = decoder->params();
        params.insert(params.end(), encoder_params.begin(), encoder_params.end());
        params.insert(params.end(), decoder_params.begin(), decoder_params.end());
        return params;
    }
};

// Placeholder for Spiking Autoencoder
class SpikingAutoencoder : public Module
{
   private:
    std::unique_ptr<Sequential> encoder;
    std::unique_ptr<Sequential> decoder;

   public:
    SpikingAutoencoder(int input_dim, int hidden_dim)
        : encoder(make_unique<Sequential>( // Encoder
              initializer_list<shared_ptr<Module>>{
                  // Layers
                  make_shared<Linear>(input_dim, hidden_dim), // hidden layer
                  make_shared<Leaky>(hidden_dim),             // Spiking layer
              })),
          decoder(make_unique<Sequential>( // Decoder
              initializer_list<shared_ptr<Module>>{
                  // Layers
                  make_shared<Linear>(hidden_dim, input_dim), // Reconstruction layer
                  // Add activation if needed (e.g., Sigmoid for [0,1] output)
              }))
    {
    }

    auto forward(const Tensor& input) -> Tensor override
    {
        Tensor encoded = encoder->forward(input);
        return decoder->forward(encoded);
    }

    auto encode(const Tensor& input) -> Tensor
    {
        return encoder->forward(input);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        Tensor grad_decoded = decoder->backward(grad_output);
        return encoder->backward(grad_decoded);
    }

    auto params() -> std::vector<Tensor*> override
    {
        std::vector<Tensor*> params;
        auto encoder_params = encoder->params();
        params.insert(params.end(), encoder_params.begin(), encoder_params.end());
        auto decoder_params = decoder->params();
        params.insert(params.end(), decoder_params.begin(), decoder_params.end());
        return params;
    }
};

static void init()
{
    // Initialization code goes here
}

// Helper function to normalize data to [0, 1]
static auto normalizeData(const Eigen::MatrixXf& data) -> Eigen::MatrixXf
{
    return Normalization::minMax(data);
}

// Helper function to apply windowing
static auto applyWindowing(const Eigen::MatrixXf& data, float window_size_sec, float overlap_ratio,
                           int sampling_rate) -> std::vector<Eigen::MatrixXf>
{
    return Windowing::slidingWindow(data, window_size_sec, overlap_ratio, sampling_rate);
}

static auto loadAndProcessAudio(const std::string& audioFilePath, float window_size_sec,
                                float overlap_ratio, int sampling_rate)
    -> std::vector<Eigen::MatrixXf>
{
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);
    Eigen::MatrixXf audioMatrix = audioSamples.transpose(); // Convert to 1xN matrix for windowing
    Eigen::MatrixXf normalizedAudio = normalizeData(audioMatrix);
    return applyWindowing(normalizedAudio, window_size_sec, overlap_ratio, sampling_rate);
}

static auto loadAndProcessEEG(const string& eegFilePath, long eegRowIndex, float window_size_sec,
                              float overlap_ratio, int sampling_rate)
    -> std::vector<Eigen::MatrixXf>
{
    auto [eegSamplesMatrix, eegInfo] = loadEEGFromMat(eegFilePath, eegRowIndex);
    Eigen::MatrixXf normalizedEEG = normalizeData(eegSamplesMatrix);
    return applyWindowing(normalizedEEG, window_size_sec, overlap_ratio, sampling_rate);
}

static void processSubject(const std::string& subjectPath, const std::string& subjectName,
                           const std::string& audioFilePath, const std::string& eegFilePath)
{
    cout << "Processing subject: " << subjectName << '\n';

    // Define experiment parameters
    const float window_size_sec = 1.5F;
    const float overlap_ratio = 0.5F;
    const int audio_sampling_rate = 44100; // Assuming 44.1 kHz for audio
    const int eeg_sampling_rate = 1024;    // Assuming 1024 Hz for EEG

    // Load, normalize, and window audio data
    std::vector<Eigen::MatrixXf> audioWindows =
        loadAndProcessAudio(audioFilePath, window_size_sec, overlap_ratio, audio_sampling_rate);
    cout << "  - Loaded and processed " << audioWindows.size() << " audio windows.\n";

    // Load, normalize, and window EEG data
    // Note: eegIndex is needed for loadEEGFromMat, but we don't have it directly here.
    // Assuming for now that we can get a representative eegRowIndex, or process all.
    // For a real experiment, this would need careful synchronization.
    long representative_eeg_index = 0; // Placeholder, needs proper handling
    std::vector<Eigen::MatrixXf> eegWindows = loadAndProcessEEG(
        eegFilePath, representative_eeg_index, window_size_sec, overlap_ratio, eeg_sampling_rate);
    cout << "  - Loaded and processed " << eegWindows.size() << " EEG windows.\n";

    // --- Autoencoder Training ---
    // Traditional Autoencoder for Audio
    if (!audioWindows.empty())
    {
        int audio_input_dim =
            (int) audioWindows[0].cols(); // Assuming each window is 1 row x N cols

        int audio_hidden_dim = audio_input_dim / 2; // Example hidden dimension

        TraditionalAutoencoder audio_ae(audio_input_dim, audio_hidden_dim);
        Adam audio_optimizer(0.001); // Example learning rate

        // Avoids the error: "reference to local variable 'params' returned"
        std::vector<Tensor*> audio_params = audio_ae.params();
        audio_optimizer.attach(audio_params);

        cout << "  - Training Traditional Autoencoder for Audio...\n";
        for (int epoch = 0; epoch < 10; ++epoch)
        { // Example: 10 epochs
            float epoch_loss = 0.0F;
            for (const Eigen::MatrixXf& window_matrix : audioWindows)
            {
                Tensor input_tensor(window_matrix);
                audio_optimizer.zero_grad(audio_params);
                Tensor output_tensor = audio_ae.forward(input_tensor);
                // Simple MSE loss for now
                Tensor loss_tensor = (output_tensor - input_tensor).square().mean();
                epoch_loss += loss_tensor.get_data()(0, 0); // Assuming scalar loss
                Tensor grad_output = (output_tensor - input_tensor) * (2.0F / input_tensor.size());
                audio_ae.backward(grad_output);
                audio_optimizer.step(audio_params);
            }
            cout << "    Epoch " << epoch << ", Loss: " << epoch_loss / (float) audioWindows.size()
                 << "\n";
        }
        cout << "  - Traditional Autoencoder training complete.\n";

        // Extract features
        std::vector<Tensor> audio_features;
        for (const Eigen::MatrixXf& window_matrix : audioWindows)
        {
            audio_features.push_back(audio_ae.encode(Tensor(window_matrix)));
        }
        cout << "  - Extracted " << audio_features.size() << " audio features.\n";
    }
    else
    {
        cout << "  - No audio windows to train Traditional Autoencoder.\n";
    }

    // Spiking Autoencoder for EEG
    if (!eegWindows.empty())
    {
        int eeg_input_dim = eegWindows[0].cols(); // Assuming each window is 1 row x N cols
        int eeg_hidden_dim = eeg_input_dim / 2;   // Example hidden dimension
        SpikingAutoencoder eeg_sae(eeg_input_dim, eeg_hidden_dim);
        Adam eeg_optimizer(0.001); // Example learning rate

        // Avoids the error: "reference to local variable 'params' returned"
        std::vector<Tensor*> eeg_params = eeg_sae.params();
        eeg_optimizer.attach(eeg_params);

        cout << "  - Training Spiking Autoencoder for EEG...\n";
        for (int epoch = 0; epoch < 10; ++epoch)
        { // Example: 10 epochs
            float epoch_loss = 0.0F;
            for (const Eigen::MatrixXf& window_matrix : eegWindows)
            {
                Tensor input_tensor(window_matrix);
                eeg_optimizer.zero_grad(eeg_params);
                Tensor output_tensor = eeg_sae.forward(input_tensor);
                // Simple MSE loss for now
                Tensor loss_tensor = (output_tensor - input_tensor).square().mean();
                epoch_loss += loss_tensor.get_data()(0, 0); // Assuming scalar loss
                Tensor grad_output =
                    (output_tensor - input_tensor) * (2.0F / (float) input_tensor.size());
                eeg_sae.backward(grad_output);
                eeg_optimizer.step(eeg_params);
            }
            cout << "    Epoch " << epoch << ", Loss: " << epoch_loss / (float) eegWindows.size()
                 << "\n";
        }
        cout << "  - Spiking Autoencoder training complete.\n";

        // Extract features
        std::vector<Tensor> eeg_features;
        eeg_features.reserve(eegWindows.size());
        for (const Eigen::MatrixXf& window_matrix : eegWindows)
        {
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
    }
    else
    {
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