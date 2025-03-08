#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include "SignalCapturer.hpp"

using namespace std;

class MockCapturer : public ICapturer
{
private:
    bool should_fail;
    string error_msg;
    bool capturing;
    thread capture_thread;

public:
    MockCapturer(bool fail = false, string err = "Mock error")
        : should_fail(fail), error_msg(err), capturing(false) {}

    bool start() override
    {
        if (should_fail)
        {
            capturing = false;
            return false;
        }

        capturing = true;

        static int i = 0;

        // Inicia a captura em uma thread separada
        capture_thread = thread([this]()
                                {
            while (capturing)
            {
                i++;
                cout << "Capturing data... " << i << endl;
            } });

        return true;
    }

    void stop() override
    {
        capturing = false;
        if (capture_thread.joinable())
        {
            capture_thread.join(); // Aguarda a thread terminar
        }
    }

    bool isCapturing() const override
    {
        return capturing;
    }

    const string &last_error() const override
    {
        return error_msg;
    }

    vector<float> getAvailableSamples() const override
    {
        return {1.0, 2.0, 3.0}; // Simula dados
    }

    ~MockCapturer()
    {
        stop(); // Garante que a thread seja finalizada no destrutor
    }
};

int main()
{
    SignalCapturer capturerManager;

    // Adiciona mock capturers
    capturerManager.addCapturer(make_unique<MockCapturer>(false, "Mock error"));
    capturerManager.addCapturer(make_unique<MockCapturer>(false, "Mock error"));

    cout << "Starting capturers..." << endl;

    bool success = capturerManager.startCapturing();

    if (success)
    {
        cout << "All capturers started successfully." << endl;
    }
    else
    {
        cout << "Some capturers failed to start!" << endl;
        cout << "Errors:\n"
             << capturerManager.getErrors() << endl;
    }

    // Aguarda alguns segundos antes de encerrar
    this_thread::sleep_for(chrono::seconds(20));

    cout << "Stopping capturers..." << endl;
    capturerManager.stopCapturing(); // Implementar este método no SignalCapturer

    return 0;
}
