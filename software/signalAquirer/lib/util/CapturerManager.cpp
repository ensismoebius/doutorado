#include "CapturerManager.hpp"

using namespace std;

bool CapturerManager::isCapturing()
{
    return capturing;
}

const string CapturerManager::getErrors() const
{
    return errors;
}

/**
 * @brief Starts all capturers in its own thread
 *
 * @return true - All started succefull
 * @return false - Some could not stop
 */
bool CapturerManager::startCapturing()
{
    vector<thread> threads;

    for (const auto &capturer : capturers)
    {

        threads.emplace_back(
            [&]()
            {
                if (!capturer->start())
                {
                    capturing = false;

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

    capturing = true;
    return capturing;
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

    capturing = false;
}
