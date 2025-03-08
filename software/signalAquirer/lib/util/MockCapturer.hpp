#ifndef MOCKCAPTURER_H
#define MOCKCAPTURER_H

#include "ICapturer.hpp"
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

using namespace std;

class MockCapturer : public ICapturer
{
private:
    bool should_fail;
    string error_msg;
    bool capturing;
    thread capture_thread;

public:
    MockCapturer(bool fail = false, string err = "Mock error");

    bool start() override;
    void stop() override;
    bool isCapturing() const override;
    const string &last_error() const override;
    const vector<float> getAvailableSamples() override;

    ~MockCapturer();
};

#endif // MOCKCAPTURER_HPP
