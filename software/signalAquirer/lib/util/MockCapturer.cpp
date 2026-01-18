#include "MockCapturer.hpp"

template <typename T>
inline T RandomRange(T min, T max)
{
    T scale = rand() / (T)RAND_MAX;
    return min + scale * (max - min);
}

MockCapturer::MockCapturer(bool fail, string err)
    : should_fail(fail), error_msg(err), capturing(false) {}

bool MockCapturer::start()
{
    if (should_fail)
    {
        capturing = false;
        return false;
    }

    this->data.resize(100);

    capturing = true;

    // // Inicia a captura em uma thread separada
    // capture_thread = thread(
    //     [this]()
    //     {
    //         int i = 0;
    //         while (capturing)
    //         {
    //             i++;
    //             cout << "Capturing data... " << i << endl;
    //             this_thread::sleep_for(chrono::seconds(2));
    //         }
    //     });

    return true;
}

void MockCapturer::stop()
{
    capturing = false;
    if (capture_thread.joinable())
    {
        capture_thread.join(); // Aguarda a thread terminar
    }
}

bool MockCapturer::isCapturing() const
{
    return capturing;
}

const string &MockCapturer::last_error() const
{
    return error_msg;
}

const vector<float> MockCapturer::getAvailableSamples()
{

    for (int i = 0; i < data.size(); ++i)
        this->data[i] = RandomRange(-1.0f, 1.0f);

    return this->data; // Simula dados
}

MockCapturer::~MockCapturer()
{
    stop(); // Garante que a thread seja finalizada no destrutor
}
