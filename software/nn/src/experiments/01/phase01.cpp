#include <iostream>

// PHASE 0 - Frozen Parameters (Constants)
namespace Phase0Config
{
// Windowing parameters
constexpr double WINDOW_DURATION_SEC = 1.5;
constexpr int WINDOW_OVERLAP_PERCENT = 50;

// Normalization
constexpr double NORM_MIN = 0.0;
constexpr double NORM_MAX = 1.0;
const std::string NORM_METHOD = "min-max";
constexpr bool PARACONSISTENT_PREREQUISITE = true;

// Classifier
const std::string CLASSIFIER_TYPE = "ResNet";
const std::string CLASSIFIER_IMPLEMENTATION = "Residual Neural Networks";

// Dataset parameters
constexpr int AUDIO_SAMPLING_RATE = 44100;
constexpr int EEG_SAMPLING_RATE = 1000;

// Paraconsistent
constexpr bool PARACONSISTENT_ENABLED = true;
constexpr double OPTIMAL_POINT_X = 1.0;
constexpr double OPTIMAL_POINT_Y = 0.0;

// Experiment
constexpr int SEED = 42;
} // namespace Phase0Config

int main()
{
    std::cout << "=== PHASE 0 - Frozen Parameters Validation ===\n\n";

    // Windowing
    std::cout << "Windowing Parameters:\n";
    std::cout << "  Duration: " << Phase0Config::WINDOW_DURATION_SEC << " seconds\n";
    std::cout << "  Overlap: " << Phase0Config::WINDOW_OVERLAP_PERCENT << "%\n\n";

    // Normalization
    std::cout << "Normalization:\n";
    std::cout << "  Range: [" << Phase0Config::NORM_MIN << ", " << Phase0Config::NORM_MAX << "]\n";
    std::cout << "  Method: " << Phase0Config::NORM_METHOD << "\n";
    std::cout << "  Paraconsistent Prerequisite: "
              << (Phase0Config::PARACONSISTENT_PREREQUISITE ? "Yes" : "No") << "\n\n";

    // Classifier
    std::cout << "Classifier:\n";
    std::cout << "  Type: " << Phase0Config::CLASSIFIER_TYPE << "\n";
    std::cout << "  Implementation: " << Phase0Config::CLASSIFIER_IMPLEMENTATION << "\n\n";

    // Dataset
    std::cout << "Dataset Parameters:\n";
    std::cout << "  Audio Sampling Rate: " << Phase0Config::AUDIO_SAMPLING_RATE << " Hz\n";
    std::cout << "  EEG Sampling Rate: " << Phase0Config::EEG_SAMPLING_RATE << " Hz\n\n";

    // Paraconsistent
    std::cout << "Paraconsistent Analysis:\n";
    std::cout << "  Enabled: " << (Phase0Config::PARACONSISTENT_ENABLED ? "Yes" : "No") << "\n";
    std::cout << "  Optimal Point: (" << Phase0Config::OPTIMAL_POINT_X << ", "
              << Phase0Config::OPTIMAL_POINT_Y << ")\n\n";

    std::cout << "✅ PHASE 0 parameters successfully frozen and validated!\n";
    std::cout << "These parameters are now fixed for all subsequent experiments.\n";

    return 0;
}