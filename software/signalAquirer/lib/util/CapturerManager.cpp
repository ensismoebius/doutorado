#include "CapturerManager.hpp"

using namespace std;

/**
 * @brief Starts all capturers in its own thread
 *
 * @return true - All started succefull
 * @return false - Some could not stop
 */
bool CapturerManager::startCapturing()
{

    vector<thread> threads;
    atomic<bool> all_success{true};

    for (const auto &capturer : capturers)
    {

        threads.emplace_back(
            [&]()
            {
                if (!capturer->start())
                {
                    all_success = false;

                    lock_guard<mutex> lock(error_mutex);
                    errors += capturer->last_error() + "\n";
                }
            });
    }

    // Wait for all threads to complete
    for (auto &t : threads)
    {
        t.join();
    }

    return all_success;
}

void CapturerManager::addCapturer(std::shared_ptr<ICapturer> capturer)
{
    capturers.push_back(capturer);
}

void CapturerManager::stopCapturing()
{
    for (auto &capturer : capturers)
    {
        capturer->stop();
    }
}
