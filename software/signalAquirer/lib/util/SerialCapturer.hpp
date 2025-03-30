#ifndef SERIAL_CAPTURER_H
#define SERIAL_CAPTURER_H

#include "ICapturer.hpp"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <iostream>

class SerialCapturer : public ICapturer
{
public:
    SerialCapturer(const std::string &device, int baudrate)
        : devicePath(device), baudRate(baudrate), fd(-1), capturing(false) {}

    bool start() override
    {
        fd = open(devicePath.c_str(), O_RDWR | O_NOCTTY);
        if (fd == -1)
        {
            lastError = "Failed to open serial port";
            return false;
        }

        struct termios options;
        tcgetattr(fd, &options);
        cfsetispeed(&options, baudRate);
        cfsetospeed(&options, baudRate);
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        tcsetattr(fd, TCSANOW, &options);

        capturing = true;
        return true;
    }

    void stop() override
    {
        if (fd != -1)
        {
            close(fd);
            fd = -1;
        }
        capturing = false;
    }

    bool isCapturing() const override
    {
        return capturing;
    }

    const std::string &last_error() const override
    {
        return lastError;
    }

    const std::vector<float> &getAvailableSamples() const override
    {
        static std::vector<float> samples;
        if (!capturing || fd == -1)
        {
            return samples;
        }

        char buffer[256];
        int bytesRead = read(fd, buffer, sizeof(buffer) - 1);
        if (bytesRead > 0)
        {
            buffer[bytesRead] = '\0';
            try
            {
                samples.push_back(std::stof(buffer));
            }
            catch (...)
            {
                lastError = "Invalid data received";
            }
        }
        return samples;
    }

    ~SerialCapturer()
    {
        stop();
    }

private:
    std::string devicePath;
    int baudRate;
    int fd;
    bool capturing;
    mutable std::string lastError;
};

#endif
