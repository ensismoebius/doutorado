#ifndef SIGNAL_CAPTURE
#define SIGNAL_CAPTURE

#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <mutex>

#include "ICapturer.hpp"

using namespace std;

class SignalCapturer
{
private:
    vector<unique_ptr<ICapturer>> capturers; // Use smart pointers to manage memory
    string errors;
    mutex error_mutex; // Mutex to protect the shared 'errors' string

public:
    SignalCapturer() = default;
    ~SignalCapturer() = default;

    void stopCapturing();
    bool startCapturing();

    void addCapturer(unique_ptr<ICapturer> capturer);
    const string &getErrors() const { return errors; } // Getter for errors
};

#endif