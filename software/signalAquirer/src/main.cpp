#include <iostream>
#include <memory>
#include <vector>

#include "../lib/util/imguiGlfw.hpp"
#include "../lib/util/AudioCapture.hpp"

// Global state for audio capture
std::unique_ptr<AudioCapture> audio_capture;
bool is_capturing = false;
std::vector<float> audio_samples;
std::string error_message;

// In your GUI rendering loop
void audio_visualization()
{
    auto samples = audio_capture->get_available_samples();
    if (!samples.empty())
    {
        ImGui::PlotLines("Audio Input", samples.data(),
                         static_cast<int>(samples.size()),
                         0, nullptr, -1.0f, 1.0f,
                         ImVec2(0, 80.0f));
    }
}

void widgets()
{
    audio_visualization();

    ImGui::Text("Hello, ImGui!");
    if (ImGui::Button("Click Me"))
    {
        std::cout << "Button clicked!" << std::endl;
    }
}

int main()
{

    // PAREI AQUI, TÔ TENTANDO DEBUGAR A CLASSE AUDIO CAPTURE E ENTENDÊ-LA
    // https://chat.deepseek.com/a/chat/s/d2eb88ef-506d-4514-ada9-13e8c8a6151d
    audio_capture = std::make_unique<AudioCapture>();
    audio_capture->start();

    ImGuiApp app("ImGui Fullscreen Frame", 800, 600);
    if (!app.initialize())
        return -1;

    // Pass a function to run
    app.run(widgets);

    audio_capture->stop();
    return 0;
}
