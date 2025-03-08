#include "SignalCapturer.hpp"

using namespace std;

bool SignalCapturer::startCapturing()
{
    vector<thread> threads;
    atomic<bool> all_success{true};

    for (const auto &capturer : capturers)
    {

        threads.emplace_back([&]()
                             {
            if (!capturer->start()) {
                all_success = false;
                
                lock_guard<mutex> lock(error_mutex);
                errors += capturer->last_error() + "\n";
            } });
    }

    // Wait for all threads to complete
    for (auto &t : threads)
    {
        t.join();
    }

    return all_success;
}

void SignalCapturer::addCapturer(unique_ptr<ICapturer> capturer)
{
    capturers.push_back(move(capturer)); // Transfer ownership
}

void SignalCapturer::stopCapturing()
{
    for (auto &capturer : capturers)
    {
        capturer->stop();
    }
}
