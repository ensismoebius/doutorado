#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include "../lib/util/CapturerManager.hpp"
#include "../lib/util/AudioCapture.hpp"
#include "../lib/util/MockCapturer.hpp"

using namespace std;
CapturerManager capturerManager;
shared_ptr<ICapturer> c0 = make_shared<MockCapturer>(false, "Mock error 0");
shared_ptr<ICapturer> c1 = make_shared<AudioCapture>();

int main()
{
    // Adiciona mock capturers
    capturerManager.addCapturer(c0);
    capturerManager.addCapturer(c1);

    bool success = capturerManager.startCapturing();

    if (success)
    {

        this_thread::sleep_for(chrono::seconds(5));

        auto data = c1->getAvailableSamples();

        for (auto const d : data)
        {
            cout << "d" << d << endl;
        }

        this_thread::sleep_for(chrono::seconds(5));

        auto data2 = c1->getAvailableSamples();

        for (auto const d : data2)
        {
            cout << "d" << d << endl;
        }

        cout << "All capturers started successfully." << endl;
    }
    else
    {
        cout << "Some capturers failed to start!" << endl;
        cout << "Errors:" << capturerManager.getErrors() << endl;
    }

    capturerManager.stopCapturing();
    return 0;
}
