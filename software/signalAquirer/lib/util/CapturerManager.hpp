#ifndef CAPTURE_MANAGER_H
#define CAPTURE_MANAGER_H

#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <mutex>

#include "ICapturer.hpp"

using namespace std;

class CapturerManager
{
private:
    vector<std::shared_ptr<ICapturer>> capturers; // Use smart pointers to manage memory
    string errors;
    mutex error_mutex; // Mutex to protect the shared 'errors' string
    atomic<bool> capturing{false};

public:
    CapturerManager() = default;
    ~CapturerManager() = default;

    bool isCapturing();
    void stopCapturing();
    bool startCapturing();

    void addCapturer(std::shared_ptr<ICapturer> capturer);
    const string getErrors() const; // Getter for errors
};

#endif